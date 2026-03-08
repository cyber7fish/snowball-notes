from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path

from ..config import SnowballConfig
from ..models import NoteMatch, SessionMemory, SessionTurn
from ..utils import normalize_text, now_utc_iso, safe_read_text, tokenize

_log = logging.getLogger(__name__)


def load_session_memory(db, conversation_id: str) -> SessionMemory:
    rows = db.fetchall(
        """
        SELECT t.turn_id, t.processed_at, t.final_decision,
               a.note_id, a.action_type, a.note_title
        FROM session_turns t
        LEFT JOIN session_note_actions a
          ON t.conversation_id = a.conversation_id AND t.turn_id = a.turn_id
        WHERE t.conversation_id = ?
        ORDER BY t.processed_at DESC
        LIMIT 20
        """,
        (conversation_id,),
    )
    memory = SessionMemory(conversation_id=conversation_id)
    for row in rows:
        memory.processed_turns.append(
            SessionTurn(
                turn_id=row["turn_id"],
                processed_at=row["processed_at"],
                final_decision=row["final_decision"],
                note_id=row.get("note_id"),
                action_type=row.get("action_type"),
                note_title=row.get("note_title"),
            )
        )
    return memory


def update_session_memory(db, conversation_id: str, turn_id: str, final_decision: str, actions: list[dict]) -> None:
    db.execute(
        """
        INSERT OR REPLACE INTO session_turns (conversation_id, turn_id, processed_at, final_decision)
        VALUES (?, ?, ?, ?)
        """,
        (conversation_id, turn_id, now_utc_iso(), final_decision),
    )
    for action in actions:
        db.execute(
            """
            INSERT OR REPLACE INTO session_note_actions (
              conversation_id, turn_id, note_id, action_type, note_title, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                turn_id,
                action["note_id"],
                action["action_type"],
                action.get("note_title"),
                now_utc_iso(),
            ),
        )


class SQLiteKnowledgeIndex:
    def __init__(self, db, config: SnowballConfig | None = None, embedding_provider=None, vector_store=None):
        self.db = db
        self.config = config
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 5) -> list[NoteMatch]:
        rows = self.db.fetchall(
            """
            SELECT note_id, title, vault_path, content_hash, metadata_json
            FROM notes
            WHERE status != 'deleted'
            """
        )
        query_norm = normalize_text(query)
        query_tokens = set(tokenize(query))
        embedding_scores = self._embedding_scores(rows, query)
        matches = []
        for row in rows:
            title = row["title"]
            title_norm = normalize_text(title)
            title_score = SequenceMatcher(None, query_norm, title_norm).ratio()
            body = ""
            note_path = Path(row["vault_path"])
            if note_path.exists():
                body = safe_read_text(note_path)[:1200]
            body_tokens = set(tokenize(body))
            body_overlap = 0.0
            if query_tokens or body_tokens:
                body_overlap = len(query_tokens & body_tokens) / max(len(query_tokens | body_tokens), 1)
            metadata = json.loads(row.get("metadata_json") or "{}")
            metadata_tokens = set()
            for key in ("tags", "topics"):
                values = metadata.get(key) or []
                if isinstance(values, list):
                    metadata_tokens.update(tokenize(" ".join(str(value) for value in values)))
            metadata_overlap_count = len(query_tokens & metadata_tokens)
            metadata_overlap = 0.0
            if query_tokens or metadata_tokens:
                metadata_overlap = metadata_overlap_count / max(len(query_tokens | metadata_tokens), 1)
            if query_norm and query_norm in title_norm:
                boost = 0.92
                if self.config is not None:
                    boost = max(boost, float(self.config.retrieval.title_match_threshold))
                title_score = max(title_score, boost)
            if metadata_overlap_count >= self._tag_min_overlap():
                metadata_overlap = max(metadata_overlap, 0.7)
            embedding_score = embedding_scores.get(row["note_id"], 0.0)
            if embedding_score >= self._embedding_threshold():
                embedding_score = max(embedding_score, self._embedding_threshold())
            similarity = self._blend_similarity(
                title_score=title_score,
                body_overlap=body_overlap,
                metadata_overlap=metadata_overlap,
                embedding_score=embedding_score,
            )
            matches.append(
                NoteMatch(
                    note_id=row["note_id"],
                    title=title,
                    vault_path=row["vault_path"],
                    similarity=similarity,
                    content_hash=row["content_hash"],
                    excerpt=body[:180],
                )
            )
        matches.sort(key=lambda item: item.similarity, reverse=True)
        return matches[:top_k]

    def load_note(self, note_id: str) -> dict:
        row = self.db.fetchone(
            "SELECT note_id, title, vault_path, content_hash, metadata_json FROM notes WHERE note_id = ?",
            (note_id,),
        )
        if row is None:
            raise KeyError(f"note {note_id} not found")
        path = Path(row["vault_path"])
        metadata = json.loads(row.get("metadata_json") or "{}")
        return {
            "note_id": row["note_id"],
            "title": row["title"],
            "vault_path": row["vault_path"],
            "content_hash": row["content_hash"],
            "content": safe_read_text(path) if path.exists() else "",
            "metadata": metadata,
        }

    def upsert_embedding(self, note_id: str) -> None:
        if self.embedding_provider is None or self.vector_store is None:
            return
        note = self.load_note(note_id)
        text = self._index_text(note["title"], note["content"], note["metadata"])
        vector = self.embedding_provider.embed(text)
        self.vector_store.upsert(
            note_id,
            vector,
            model_name=self.embedding_provider.model_name,
            content_hash=note["content_hash"],
        )

    def _embedding_scores(self, rows: list[dict], query: str) -> dict[str, float]:
        if self.embedding_provider is None or self.vector_store is None or not rows:
            return {}
        try:
            texts: list[str] = []
            upserts: list[tuple[str, str]] = []
            for row in rows:
                existing = getattr(self.vector_store, "get_row", lambda note_id: None)(row["note_id"])
                if (
                    existing is not None
                    and existing.get("embedding_model") == self.embedding_provider.model_name
                    and existing.get("content_hash") == row["content_hash"]
                ):
                    continue
                note_path = Path(row["vault_path"])
                content = safe_read_text(note_path) if note_path.exists() else ""
                metadata = json.loads(row.get("metadata_json") or "{}")
                texts.append(self._index_text(row["title"], content, metadata))
                upserts.append((row["note_id"], row["content_hash"]))
            if texts:
                vectors = self.embedding_provider.embed_batch(texts)
                for (note_id, content_hash), vector in zip(upserts, vectors):
                    self.vector_store.upsert(
                        note_id,
                        vector,
                        model_name=self.embedding_provider.model_name,
                        content_hash=content_hash,
                    )
            query_vector = self.embedding_provider.embed(query)
            top_k = self.config.retrieval.embedding_top_k if self.config is not None else 5
            return {
                item.note_id: item.similarity
                for item in self.vector_store.search(query_vector, top_k=top_k)
            }
        except Exception:
            _log.warning("embedding search failed, falling back to text-only retrieval", exc_info=True)
            return {}

    def _index_text(self, title: str, content: str, metadata: dict) -> str:
        strategy = self.config.embedding.index_text_strategy if self.config is not None else "title_plus_summary"
        tags = " ".join(str(value) for value in metadata.get("tags", []) if value)
        topics = " ".join(str(value) for value in metadata.get("topics", []) if value)
        summary = content[:800]
        if strategy == "title_only":
            return title
        return f"{title}\n{tags}\n{topics}\n{summary}".strip()

    def _blend_similarity(
        self,
        *,
        title_score: float,
        body_overlap: float,
        metadata_overlap: float,
        embedding_score: float,
    ) -> float:
        if self.embedding_provider is None or self.vector_store is None:
            return round((title_score * 0.75) + (body_overlap * 0.25), 3)
        similarity = (
            (title_score * 0.55)
            + (body_overlap * 0.15)
            + (metadata_overlap * 0.10)
            + (embedding_score * 0.20)
        )
        if title_score >= self._title_match_threshold():
            similarity = max(similarity, title_score)
        return round(min(max(similarity, 0.0), 1.0), 3)

    def _tag_min_overlap(self) -> int:
        if self.config is None:
            return 2
        return max(int(self.config.retrieval.tag_min_overlap), 1)

    def _embedding_threshold(self) -> float:
        if self.config is None:
            return 0.80
        return float(self.config.retrieval.embedding_threshold)

    def _title_match_threshold(self) -> float:
        if self.config is None:
            return 0.85
        return float(self.config.retrieval.title_match_threshold)
