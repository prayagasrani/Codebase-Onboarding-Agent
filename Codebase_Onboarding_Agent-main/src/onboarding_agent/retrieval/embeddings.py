"""Embedding providers.

``HashingEmbedder`` is a deterministic, dependency-free feature-hashing
embedder (TF over hashed token buckets, L2-normalized). It runs offline and
keeps CI hermetic. ``SentenceTransformerEmbedder`` is the production option.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np

from onboarding_agent.retrieval.tokenize import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Feature-hashing bag-of-tokens embedder (offline, deterministic)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return idx, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, count in counts.items():
                idx, sign = self._bucket(tok)
                out[row, idx] += sign * (1.0 + math.log(count))
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:
    """Dense embeddings via sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def build_embedder(provider: str, dim: int, model_name: str) -> Embedder:
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    return HashingEmbedder(dim)
