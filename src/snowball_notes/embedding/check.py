from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..utils import new_id, now_utc_iso, sha256_text
from .dashscope import DashScopeEmbeddingProvider
from .local import LocalHashEmbeddingProvider
from .sqlite_blob import SQLiteBlobVectorStore
from .sqlite_vec import SqliteVecStore
from .voyage import VoyageEmbeddingProvider


DEFAULT_PROBE_TEXT = (
    "Agent runtime guardrails keep tool calls side-effect free until commit validation."
)


def run_embedding_check(
    config,
    db,
    *,
    provider_override: str | None = None,
    vector_store_override: str | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    effective_config = deepcopy(config)
    if provider_override:
        effective_config.embedding.provider = provider_override
    if vector_store_override:
        effective_config.embedding.vector_store = vector_store_override

    provider = _build_provider(effective_config)
    vector_store = _build_vector_store(effective_config, db)
    probe_text = (text or DEFAULT_PROBE_TEXT).strip() or DEFAULT_PROBE_TEXT

    vector = provider.embed(probe_text)
    probe_id = new_id("embedding_probe")
    content_hash = sha256_text(probe_text)
    vector_store.upsert(
        probe_id,
        vector,
        model_name=provider.model_name,
        content_hash=content_hash,
    )
    results = vector_store.search(vector, top_k=1)
    vector_store.delete(probe_id)
    db.commit()

    top_result = results[0] if results else None
    similarity = float(top_result.similarity) if top_result is not None else 0.0
    roundtrip_ok = (
        top_result is not None
        and top_result.note_id == probe_id
        and similarity >= 0.999
    )
    api_key_env = _provider_api_key_env(effective_config)
    return {
        "ok": bool(vector) and roundtrip_ok,
        "provider": effective_config.embedding.provider,
        "model_name": provider.model_name,
        "vector_store": effective_config.embedding.vector_store,
        "probe_text": probe_text,
        "dimensions": len(vector),
        "roundtrip_ok": roundtrip_ok,
        "roundtrip_similarity": similarity if top_result is not None else None,
        "timestamp": now_utc_iso(),
        "api_key_env": api_key_env,
        "api_key_required": effective_config.embedding.provider in {"voyage", "dashscope"},
        "api_base_url": _provider_api_base_url(effective_config),
    }


def _build_provider(config):
    if config.embedding.provider == "dashscope":
        return DashScopeEmbeddingProvider(config)
    if config.embedding.provider == "voyage":
        return VoyageEmbeddingProvider(config)
    return LocalHashEmbeddingProvider(config)


def _build_vector_store(config, db):
    if config.embedding.vector_store == "sqlite_vec":
        return SqliteVecStore(db)
    return SQLiteBlobVectorStore(db)


def _provider_api_key_env(config) -> str | None:
    if config.embedding.provider == "dashscope":
        return config.embedding.dashscope_api_key_env
    if config.embedding.provider == "voyage":
        return config.embedding.api_key_env
    return None


def _provider_api_base_url(config) -> str | None:
    if config.embedding.provider == "dashscope":
        return config.embedding.dashscope_api_base_url
    if config.embedding.provider == "voyage":
        return config.embedding.api_base_url
    return None
