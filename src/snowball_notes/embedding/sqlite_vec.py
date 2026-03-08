from __future__ import annotations

from .sqlite_blob import SQLiteBlobVectorStore


class SqliteVecStore(SQLiteBlobVectorStore):
    """Fallback implementation until sqlite-vec is wired in."""
