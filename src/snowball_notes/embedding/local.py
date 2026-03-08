from __future__ import annotations

import hashlib
import math

from ..config import SnowballConfig
from ..utils import normalize_text, tokenize


class LocalHashEmbeddingProvider:
    def __init__(self, config: SnowballConfig):
        self.config = config
        self.model_name = config.embedding.local_model
        self.dimensions = max(int(config.embedding.local_dimensions), 32)

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            fallback = normalize_text(text)
            if fallback:
                tokens = [fallback[index : index + 3] for index in range(0, len(fallback), 3)]
        if not tokens:
            return [0.0] * self.dimensions
        vector = [0.0] * self.dimensions
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vector[index] += sign * weight
        return _normalize(vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0:
        return vector
    return [component / magnitude for component in vector]
