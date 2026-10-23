"""Rerankers applied on top of fused retrieval results."""

from __future__ import annotations

import json

from onboarding_agent.logging import get_logger
from onboarding_agent.models import SearchResult
from onboarding_agent.retrieval.tokenize import tokenize

log = get_logger(__name__)


class NoopReranker:
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        return results[:top_k]


class LexicalOverlapReranker:
    """Cheap deterministic reranker: token overlap with a symbol-name boost.

    Serves as offline stand-in and eval baseline for the LLM reranker.
    """

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return results[:top_k]
        rescored: list[SearchResult] = []
        for r in results:
            overlap = len(q_tokens & set(tokenize(r.chunk.text))) / len(q_tokens)
            name_boost = 0.0
            if r.chunk.symbol_name:
                name_boost = 0.5 * len(q_tokens & set(tokenize(r.chunk.symbol_name))) / len(q_tokens)
            rescored.append(SearchResult(r.chunk, r.score + overlap + name_boost, source="rerank"))
        rescored.sort(key=lambda r: -r.score)
        return rescored[:top_k]


class LLMReranker:
    """LLM-as-reranker: orders candidate chunks by relevance to the query."""

    def __init__(self, llm) -> None:  # llm: onboarding_agent.agent.llm.LLM
        self._llm = llm
        self._fallback = LexicalOverlapReranker()

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        candidates = results[: max(top_k * 3, 15)]
        listing = "\n\n".join(
            f"[{i}] {r.chunk.citation} ({r.chunk.symbol_name or 'module'})\n{r.chunk.text[:600]}"
            for i, r in enumerate(candidates)
        )
        prompt = (
            "Rank the following code chunks by relevance to the developer question.\n"
            f"Question: {query}\n\nChunks:\n{listing}\n\n"
            'Return ONLY a JSON array of chunk indexes, most relevant first, e.g. [2,0,5].'
        )
        try:
            raw = self._llm.complete(prompt, max_tokens=300)
            start, end = raw.find("["), raw.rfind("]")
            order = json.loads(raw[start:end + 1])
            seen: set[int] = set()
            ranked: list[SearchResult] = []
            for idx in order:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                    seen.add(idx)
                    r = candidates[idx]
                    ranked.append(SearchResult(r.chunk, float(len(candidates) - len(ranked)), "rerank"))
            for i, r in enumerate(candidates):
                if i not in seen:
                    ranked.append(SearchResult(r.chunk, 0.0, "rerank"))
            return ranked[:top_k]
        except Exception as exc:  # noqa: BLE001
            log.warning("llm_rerank_failed_falling_back", error=str(exc))
            return self._fallback.rerank(query, results, top_k)
