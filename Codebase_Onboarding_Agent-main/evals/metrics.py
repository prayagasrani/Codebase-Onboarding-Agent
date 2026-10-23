"""Evaluation metrics: Recall@K, MRR, correctness, citation accuracy, hallucination."""

from __future__ import annotations


def file_of(ref: str) -> str:
    return ref.split(":")[0].replace("\\", "/")


def recall_at_k(retrieved_files: list[str], expected_files: list[str], k: int) -> float:
    """1.0 if any expected file appears in the top-k retrieved files."""
    top = {f.replace("\\", "/") for f in retrieved_files[:k]}
    return 1.0 if any(e in top for e in expected_files) else 0.0


def reciprocal_rank(retrieved_files: list[str], expected_files: list[str]) -> float:
    for rank, f in enumerate(retrieved_files, start=1):
        if f.replace("\\", "/") in expected_files:
            return 1.0 / rank
    return 0.0


def keyword_correctness(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    low = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in low) / len(keywords)


def citation_accuracy(verification: dict) -> float:
    """Share of emitted citations that are supported by gathered evidence."""
    return float(verification.get("support_rate", 1.0))


def is_hallucinated(verification: dict, answer: str) -> bool:
    """An answer hallucinates if it emits any unsupported citation."""
    _ = answer
    return bool(verification.get("unsupported"))


def aggregate(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
