from __future__ import annotations

from typing import Protocol

from .provider import SearchResult


class VectorStore(Protocol):
    def upsert(self, note_id: str, vector: list[float], *, model_name: str, content_hash: str) -> None: ...

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]: ...

    def delete(self, note_id: str) -> None: ...
