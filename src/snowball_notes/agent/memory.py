from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

from ..models import NoteMatch, SessionMemory, SessionTurn
from ..utils import normalize_text, now_utc_iso, safe_read_text, tokenize


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
    def __init__(self, db):
        self.db = db

    def search(self, query: str, top_k: int = 5) -> list[NoteMatch]:
        rows = self.db.fetchall(
            "SELECT note_id, title, vault_path, content_hash FROM notes WHERE status != 'deleted'"
        )
        query_norm = normalize_text(query)
        query_tokens = set(tokenize(query))
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
            overlap = 0.0
            if query_tokens or body_tokens:
                overlap = len(query_tokens & body_tokens) / max(len(query_tokens | body_tokens), 1)
            if query_norm and query_norm in title_norm:
                title_score = max(title_score, 0.92)
            similarity = round((title_score * 0.75) + (overlap * 0.25), 3)
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

