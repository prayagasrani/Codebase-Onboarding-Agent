"""CLI: index, refresh, ask, onboard, serve, eval."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onboarding-agent",
                                     description="Codebase Onboarding Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index a local repo path or GitHub URL")
    p_index.add_argument("repo")

    sub.add_parser("refresh", help="Incrementally re-index the current repository")

    p_ask = sub.add_parser("ask", help="Ask a question about the indexed repository")
    p_ask.add_argument("question")
    p_ask.add_argument("--user", default="default")

    sub.add_parser("onboard", help="Generate the 'Onboard Me' guide")
    sub.add_parser("serve", help="Run the FastAPI server")

    p_eval = sub.add_parser("eval", help="Run the retrieval/answer benchmark")
    p_eval.add_argument("--dataset", default="evals/benchmark.jsonl")
    p_eval.add_argument("--output", default="eval_results")
    p_eval.add_argument("--no-llm", action="store_true", help="Skip LLM-dependent metrics")

    args = parser.parse_args(argv)

    if args.command == "serve":
        from onboarding_agent.api.main import run

        run()
        return 0

    if args.command == "eval":
        from evals.run_eval import run_evaluation  # repo-level module

        summary = run_evaluation(dataset_path=args.dataset, output_dir=args.output,
                                 use_llm=not args.no_llm)
        print(json.dumps(summary, indent=2, default=str))
        return 0

    from onboarding_agent.service import AppService

    service = AppService()
    if args.command == "index":
        print(json.dumps(service.index_repo(args.repo), indent=2))
    elif args.command == "refresh":
        print(json.dumps(service.refresh_index().to_dict(), indent=2))
    elif args.command == "ask":
        result = service.ask(args.question, user_id=args.user)
        print(result["answer"])
        if result["citations"]:
            print("\nSources: " + ", ".join(c["ref"] for c in result["citations"]))
        print(f"\n[route={result['route']} latency={result['latency_s']}s "
              f"citation_support={result['verification'].get('support_rate', 1.0):.0%}]")
    elif args.command == "onboard":
        from onboarding_agent.onboarding.generator import OnboardingGenerator

        result = OnboardingGenerator(service).generate()
        print(result["guide"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
