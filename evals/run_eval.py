"""Repository-specific benchmark runner.

Indexes this repository itself, then measures:
  1. Retrieval quality (Recall@K, MRR) for each config:
     vector | bm25 | hybrid | hybrid_rrf | hybrid_rrf_rerank
  2. End-to-end answers: correctness (keyword coverage), citation accuracy,
     hallucination rate (unsupported citations), latency.
  3. Memory effect: does a follow-up conversation see the prior context?

Exits non-zero (via --check) when quality falls below thresholds — wired into CI.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from evals.metrics import (
    aggregate,
    citation_accuracy,
    is_hallucinated,
    keyword_correctness,
    recall_at_k,
    reciprocal_rank,
)
from onboarding_agent.config import Settings
from onboarding_agent.retrieval.retriever import MODES
from onboarding_agent.service import AppService

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_THRESHOLDS = {
    "retrieval_recall_at_10": 0.75,   # hybrid_rrf, file-level
    "retrieval_mrr": 0.45,            # hybrid_rrf
    "answer_correctness": 0.50,       # mean keyword coverage
    "citation_accuracy": 0.90,        # mean supported-citation rate
    "hallucination_rate_max": 0.10,   # max share of answers with unsupported citations
}


def load_dataset(path: str | Path) -> list[dict]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def build_service(tmp_index_dir: Path) -> AppService:
    settings = Settings(
        groq_api_key="",  # retrieval eval is LLM-independent; agent runs extractive
        github_token="",
        github_repo="",  # keep the benchmark hermetic — no live GitHub calls
        database_url="sqlite:///:memory:",
        index_dir=tmp_index_dir,
        embedding_provider="hash",
        reranker="llm",  # resolves to LexicalOverlapReranker offline
    )
    service = AppService(settings)
    service.index.build(REPO_ROOT / "src")
    service.retriever = service._build_retriever()  # rebind to the freshly built stores
    # Benchmark expected_files are repo-root relative; index paths are src-relative.
    return service


def eval_retrieval(service: AppService, dataset: list[dict], k: int = 10) -> dict:
    results: dict = {}
    for mode in MODES:
        recalls, mrrs, latencies = [], [], []
        for item in dataset:
            expected = [f.removeprefix("src/") for f in item["expected_files"]]
            t0 = time.perf_counter()
            hits = service.retriever.search(item["question"], top_k=k, mode=mode)
            latencies.append(time.perf_counter() - t0)
            files = [h.chunk.file_path for h in hits]
            recalls.append(recall_at_k(files, expected, k))
            mrrs.append(reciprocal_rank(files, expected))
        results[mode] = {
            f"recall@{k}": round(aggregate(recalls), 3),
            "mrr": round(aggregate(mrrs), 3),
            "p50_latency_ms": round(statistics.median(latencies) * 1000, 1),
        }
    return results


def eval_answers(service: AppService, dataset: list[dict]) -> dict:
    correctness, cite_acc, halluc, latencies = [], [], [], []
    per_item = []
    for item in dataset:
        result = service.ask(item["question"], user_id="eval-user")
        expected = [f.removeprefix("src/") for f in item["expected_files"]]
        cited_files = [c["ref"].split(":")[0] for c in result["citations"] if c["kind"] == "code"]
        correct = keyword_correctness(result["answer"], item.get("keywords", []))
        # Citing at least one expected file also counts toward correctness signal.
        file_hit = 1.0 if any(f in expected for f in cited_files) else 0.0
        score = max(correct, file_hit)
        correctness.append(score)
        cite_acc.append(citation_accuracy(result["verification"]))
        halluc.append(1.0 if is_hallucinated(result["verification"], result["answer"]) else 0.0)
        latencies.append(result["latency_s"])
        per_item.append({"id": item["id"], "correctness": round(score, 2),
                         "citation_accuracy": round(cite_acc[-1], 2),
                         "hallucinated": bool(halluc[-1]), "latency_s": result["latency_s"]})
    return {
        "answer_correctness": round(aggregate(correctness), 3),
        "citation_accuracy": round(aggregate(cite_acc), 3),
        "hallucination_rate": round(aggregate(halluc), 3),
        "p50_latency_s": round(statistics.median(latencies), 3) if latencies else 0.0,
        "per_item": per_item,
    }


def eval_memory(service: AppService) -> dict:
    """Does memory carry context into subsequent conversations?"""
    user = "memory-eval-user"
    q1 = "How does reciprocal rank fusion combine vector and bm25 rankings?"
    q2 = "Can you remind me what we discussed about rank fusion?"
    service.ask(q1, user_id=user)
    ctx = service.memory.familiarity_summary(user, q2)
    recalled = len(ctx["previously_discussed"]) > 0
    concepts_tracked = service.memory.familiarity(user, ["fusion", "rank"])
    second = service.ask(q2, user_id=user)
    fresh_ctx = service.memory.familiarity_summary("fresh-user", q2)
    return {
        "prior_interaction_recalled": recalled,
        "concepts_tracked": {k: round(v, 2) for k, v in concepts_tracked.items()},
        "memory_route_used": second["route"] == "memory",
        "fresh_user_has_no_context": len(fresh_ctx["previously_discussed"]) == 0,
        "memory_improves_context": recalled and len(fresh_ctx["previously_discussed"]) == 0,
    }


def check_thresholds(summary: dict, thresholds: dict) -> list[str]:
    failures = []
    rrf = summary["retrieval"]["hybrid_rrf"]
    if rrf["recall@10"] < thresholds["retrieval_recall_at_10"]:
        failures.append(f"hybrid_rrf recall@10 {rrf['recall@10']} < {thresholds['retrieval_recall_at_10']}")
    if rrf["mrr"] < thresholds["retrieval_mrr"]:
        failures.append(f"hybrid_rrf MRR {rrf['mrr']} < {thresholds['retrieval_mrr']}")
    ans = summary["answers"]
    if ans["answer_correctness"] < thresholds["answer_correctness"]:
        failures.append(f"answer correctness {ans['answer_correctness']} < {thresholds['answer_correctness']}")
    if ans["citation_accuracy"] < thresholds["citation_accuracy"]:
        failures.append(f"citation accuracy {ans['citation_accuracy']} < {thresholds['citation_accuracy']}")
    if ans["hallucination_rate"] > thresholds["hallucination_rate_max"]:
        failures.append(f"hallucination rate {ans['hallucination_rate']} > {thresholds['hallucination_rate_max']}")
    if not summary["memory"]["memory_improves_context"]:
        failures.append("memory did not improve subsequent conversation context")
    return failures


def run_evaluation(dataset_path: str = "evals/benchmark.jsonl", output_dir: str = "eval_results",
                   use_llm: bool = False, thresholds: dict | None = None) -> dict:
    _ = use_llm  # the offline path is authoritative for CI; LLM judging is additive
    thresholds = thresholds or DEFAULT_THRESHOLDS
    dataset = load_dataset(REPO_ROOT / dataset_path if not Path(dataset_path).is_absolute()
                           else dataset_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    service = build_service(out / "index")
    summary = {
        "dataset": str(dataset_path),
        "items": len(dataset),
        "index_stats": service.index.stats(),
        "retrieval": eval_retrieval(service, dataset),
        "answers": eval_answers(service, dataset),
        "memory": eval_memory(service),
        "thresholds": thresholds,
    }
    summary["failures"] = check_thresholds(summary, thresholds)
    summary["passed"] = not summary["failures"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the onboarding-agent benchmark")
    parser.add_argument("--dataset", default="evals/benchmark.jsonl")
    parser.add_argument("--output", default="eval_results")
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if quality falls below thresholds (CI gate)")
    args = parser.parse_args()

    summary = run_evaluation(dataset_path=args.dataset, output_dir=args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != "answers"} |
                     {"answers": {k: v for k, v in summary["answers"].items() if k != "per_item"}},
                     indent=2, default=str))
    if args.check and not summary["passed"]:
        print("EVAL GATE FAILED:", *summary["failures"], sep="\n  - ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
