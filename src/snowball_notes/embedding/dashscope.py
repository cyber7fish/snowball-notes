from __future__ import annotations

import json
import os
from urllib import error, request

from ..config import SnowballConfig


class DashScopeEmbeddingProvider:
    def __init__(self, config: SnowballConfig):
        self.config = config
        self.model_name = config.embedding.dashscope_model
        self.dimensions = int(config.embedding.dashscope_dimensions)
        api_key = os.environ.get(config.embedding.dashscope_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"missing embedding API key env: {config.embedding.dashscope_api_key_env}"
            )
        self.api_key = api_key
        self.api_base_url = config.embedding.dashscope_api_base_url

    def embed(self, text: str) -> list[float]:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
            "dimensions": self.dimensions,
        }
        raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.api_base_url,
            data=raw_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.agent.request_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"dashscope embedding request failed: {exc.code} {detail}".strip()) from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"dashscope embedding request failed: {exc}") from exc
        vectors = []
        for item in payload.get("data", []):
            embedding = item.get("embedding")
            if isinstance(embedding, list):
                vectors.append([float(value) for value in embedding])
        return vectors
