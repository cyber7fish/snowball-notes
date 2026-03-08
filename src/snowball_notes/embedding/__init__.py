from __future__ import annotations

from ..config import SnowballConfig
from .check import run_embedding_check
from .dashscope import DashScopeEmbeddingProvider
from .local import LocalHashEmbeddingProvider
from .sqlite_blob import SQLiteBlobVectorStore
from .sqlite_vec import SqliteVecStore
from .voyage import VoyageEmbeddingProvider


def build_embedding_provider(config: SnowballConfig):
    if config.embedding.provider == "dashscope":
        return DashScopeEmbeddingProvider(config)
    if config.embedding.provider == "voyage":
        return VoyageEmbeddingProvider(config)
    return LocalHashEmbeddingProvider(config)


def build_vector_store(config: SnowballConfig, db):
    if config.embedding.vector_store == "sqlite_vec":
        return SqliteVecStore(db)
    return SQLiteBlobVectorStore(db)


__all__ = [
    "DashScopeEmbeddingProvider",
    "LocalHashEmbeddingProvider",
    "SQLiteBlobVectorStore",
    "SqliteVecStore",
    "VoyageEmbeddingProvider",
    "build_embedding_provider",
    "build_vector_store",
    "run_embedding_check",
]
