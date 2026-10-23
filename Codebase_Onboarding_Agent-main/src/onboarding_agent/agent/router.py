"""Query/intent routing.

A deterministic keyword router handles the common cases (and the offline
path); when an LLM is available and the heuristic is not confident, the LLM
refines the route. The route restricts which tools the agent loop exposes so
we do not call every source on every query.
"""

from __future__ import annotations

import re

from onboarding_agent.agent.prompts import ROUTER_PROMPT
from onboarding_agent.logging import get_logger

log = get_logger(__name__)

ROUTES = ("code", "github", "memory", "mixed")

_GITHUB_PATTERNS = [
    r"\bpull request", r"\bprs?\b", r"\bpr\s*#?\d+", r"\bissues?\b", r"\bcommits?\b",
    r"\bbranch(es)?\b", r"\bci\b", r"\bpipeline", r"\bbuild (status|failing|passing)",
    r"\bmerged?\b", r"\brecently (changed|updated|modified)", r"\bwho (changed|wrote|added)",
    r"\blast (change|commit|update)", r"\bin flight\b", r"\bin progress\b",
    r"\bcurrently (happening|being worked)", r"\breview\b", r"\bopen\b.*\b(pr|issue)",
]
_MEMORY_PATTERNS = [
    r"\bwhat did i\b", r"\bdid i ask\b", r"\bremind me\b", r"\bwe (discussed|talked)\b",
    r"\bearlier\b.*\b(asked|said|explained)", r"\bmy (previous|past|earlier) question",
    r"\balready explained\b", r"\blast time\b",
]
_CODE_PATTERNS = [
    r"\bhow (does|do|is)\b", r"\bwhere is\b", r"\bimplemented?\b", r"\bfunction\b",
    r"\bclass\b", r"\bmodule\b", r"\barchitecture\b", r"\bentry ?point", r"\bdepends? on\b",
    r"\bcalled? (by|from)\b", r"\bauth", r"\bdata flow\b", r"\bconfig", r"\btest",
]


def _score(patterns: list[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text))


class QueryRouter:
    def __init__(self, llm=None):
        self._llm = llm

    def route(self, question: str) -> str:
        q = question.lower()
        gh = _score(_GITHUB_PATTERNS, q)
        mem = _score(_MEMORY_PATTERNS, q)
        code = _score(_CODE_PATTERNS, q)

        if mem > 0 and mem >= gh and mem >= code:
            return "memory"
        if gh > 0 and code > 0:
            return "mixed"
        if gh > 0 and code == 0:
            return "github"
        if code > 0:
            return "code"

        # Ambiguous: let the LLM decide when available, else default to code.
        if self._llm is not None:
            try:
                raw = self._llm.complete(ROUTER_PROMPT.format(question=question), max_tokens=10)
                candidate = raw.strip().lower().split()[0] if raw.strip() else ""
                if candidate in ROUTES:
                    log.info("router_llm_refined", route=candidate)
                    return candidate
            except Exception as exc:  # noqa: BLE001
                log.warning("router_llm_failed", error=str(exc))
        return "code"
