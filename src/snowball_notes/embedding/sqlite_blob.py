from __future__ import annotations

import json
from math import sqrt

from ..utils import now_utc_iso
from .provider import SearchResult


class SQLiteBlobVectorStore:
    def __init__(self, db):
        self.db = db

    def upsert(self, note_id: str, vector: list[float], *, model_name: str, content_hash: str) -> None:
        payload = json.dumps([float(value) for value in vector], separators=(",", ":")).encode("utf-8")
        self.db.execute(
            """
            INSERT INTO note_embeddings (note_id, embedding_model, embedding_vector, content_hash, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(note_id)
            DO UPDATE SET embedding_model = excluded.embedding_model,
                          embedding_vector = excluded.embedding_vector,
                          content_hash = excluded.content_hash,
                          updated_at = excluded.updated_at
            """,
            (note_id, model_name, payload, content_hash, now_utc_iso()),
        )

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        rows = self.db.fetchall(
            """
            SELECT note_id, embedding_vector
            FROM note_embeddings
            WHERE embedding_vector IS NOT NULL
            """
        )
        scored = []
        for row in rows:
            vector = _decode_vector(row["embedding_vector"])
            if not vector:
                continue
            scored.append(SearchResult(note_id=row["note_id"], similarity=round(_cosine(query_vector, vector), 6)))
        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:top_k]

    def delete(self, note_id: str) -> None:
        self.db.execute("DELETE FROM note_embeddings WHERE note_id = ?", (note_id,))

    def get_row(self, note_id: str) -> dict | None:
        return self.db.fetchone(
            """
            SELECT note_id, embedding_model, content_hash, updated_at
            FROM note_embeddings
            WHERE note_id = ?
            """,
            (note_id,),
        )


def _decode_vector(raw) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = str(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [float(value) for value in payload]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(length))
    left_norm = sqrt(sum(value * value for value in left[:length]))
    right_norm = sqrt(sum(value * value for value in right[:length]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
