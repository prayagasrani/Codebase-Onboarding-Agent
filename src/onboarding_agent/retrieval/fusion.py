"""Reciprocal Rank Fusion."""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists: score(d) = sum_i w_i / (k + rank_i(d)).

    Ranks are 1-based; documents missing from a ranking contribute nothing.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
