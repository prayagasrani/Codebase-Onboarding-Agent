"""Vector index: FAISS when available, exact NumPy search otherwise.

Both paths use inner product on normalized vectors (cosine). The store
persists vectors + chunk-id mapping so indexes survive restarts and support
incremental add/remove.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from onboarding_agent.logging import get_logger

log = get_logger(__name__)

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[str] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._index = faiss.IndexFlatIP(dim) if _HAS_FAISS else None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        if len(chunk_ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        self._ids.extend(chunk_ids)
        self._vectors = np.vstack([self._vectors, vectors])
        if self._index is not None:
            self._index.add(vectors)

    def remove(self, chunk_ids: set[str]) -> None:
        if not chunk_ids:
            return
        keep = [i for i, cid in enumerate(self._ids) if cid not in chunk_ids]
        self._ids = [self._ids[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        if self._index is not None:
            self._index.reset()
            if len(self._ids):
                self._index.add(self._vectors)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(top_k, len(self._ids))
        if self._index is not None:
            scores, idxs = self._index.search(q, k)
            return [(self._ids[i], float(s)) for i, s in zip(idxs[0], scores[0], strict=True) if i >= 0]
        sims = (self._vectors @ q[0]).astype(float)
        order = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in order]

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._vectors)
        (directory / "vector_ids.json").write_text(json.dumps(self._ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, dim: int) -> "VectorStore":
        store = cls(dim)
        vec_path = directory / "vectors.npy"
        ids_path = directory / "vector_ids.json"
        if vec_path.exists() and ids_path.exists():
            vectors = np.load(vec_path)
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            store.add(ids, vectors)
        return store
