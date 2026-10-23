"""Hybrid retriever: vector + BM25 + RRF (+ optional reranker).

Supports explicit modes for the evaluation matrix:
  vector | bm25 | hybrid | hybrid_rrf | hybrid_rrf_rerank
("hybrid" = normalized score interpolation; "hybrid_rrf" = rank fusion.)
"""

from __future__ import annotations

from onboarding_agent.models import CodeChunk, SearchResult
from onboarding_agent.retrieval.bm25_store import BM25Store
from onboarding_agent.retrieval.embeddings import Embedder
from onboarding_agent.retrieval.fusion import reciprocal_rank_fusion
from onboarding_agent.retrieval.vector_store import VectorStore

MODES = ("vector", "bm25", "hybrid", "hybrid_rrf", "hybrid_rrf_rerank")


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        chunks: dict[str, CodeChunk],
        rrf_k: int = 60,
        reranker=None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.reranker = reranker

    def search(self, query: str, top_k: int = 10, mode: str = "hybrid_rrf") -> list[SearchResult]:
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {MODES}")
        fetch_k = max(top_k * 3, 20)

        vec_hits: list[tuple[str, float]] = []
        bm_hits: list[tuple[str, float]] = []
        if mode != "bm25":
            q_vec = self.embedder.embed([query])[0]
            vec_hits = self.vector_store.search(q_vec, fetch_k)
        if mode != "vector":
            bm_hits = self.bm25_store.search(query, fetch_k)

        if mode == "vector":
            results = self._to_results(vec_hits, "vector")
        elif mode == "bm25":
            results = self._to_results(bm_hits, "bm25")
        elif mode == "hybrid":
            results = self._interpolate(vec_hits, bm_hits)
        else:  # hybrid_rrf / hybrid_rrf_rerank
            fused = reciprocal_rank_fusion(
                [[cid for cid, _ in vec_hits], [cid for cid, _ in bm_hits]], k=self.rrf_k
            )
            results = self._to_results(fused, "hybrid")

        if mode == "hybrid_rrf_rerank" and self.reranker is not None:
            return self.reranker.rerank(query, results, top_k)
        return results[:top_k]

    def _to_results(self, hits: list[tuple[str, float]], source: str) -> list[SearchResult]:
        return [
            SearchResult(self.chunks[cid], score, source)
            for cid, score in hits
            if cid in self.chunks
        ]

    def _interpolate(
        self, vec_hits: list[tuple[str, float]], bm_hits: list[tuple[str, float]], alpha: float = 0.5
    ) -> list[SearchResult]:
        def normalize(hits: list[tuple[str, float]]) -> dict[str, float]:
            if not hits:
                return {}
            lo = min(s for _, s in hits)
            hi = max(s for _, s in hits)
            span = (hi - lo) or 1.0
            return {cid: (s - lo) / span for cid, s in hits}

        v, b = normalize(vec_hits), normalize(bm_hits)
        combined = {cid: alpha * v.get(cid, 0.0) + (1 - alpha) * b.get(cid, 0.0) for cid in {*v, *b}}
        ordered = sorted(combined.items(), key=lambda kv: -kv[1])
        return self._to_results(ordered, "hybrid")
