"""Citation extraction and verification.

Answers cite sources inline like [auth/service.py:42-91], [PR #421],
[commit abc1234]. We extract those, classify them, and verify each against
the evidence gathered by tools — an unsupported citation counts toward the
hallucination metric in evals.
"""

from __future__ import annotations

import re

from onboarding_agent.models import Citation

_CITE_RE = re.compile(r"\[([^\[\]]{1,120})\]")
# LLMs (especially open models) emit typographic dashes in line ranges, e.g. "72‑84".
_DASH_TRANSLATION = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def _normalize_dashes(text: str) -> str:
    return text.translate(_DASH_TRANSLATION)
_CODE_REF_RE = re.compile(r"^[\w./\\-]+:\d+(?:-\d+)?$")
_PR_REF_RE = re.compile(r"^PR\s*#\d+$", re.IGNORECASE)
_ISSUE_REF_RE = re.compile(r"^issue\s*#\d+$", re.IGNORECASE)
_COMMIT_REF_RE = re.compile(r"^commit\s+[0-9a-f]{6,40}$", re.IGNORECASE)


def classify_ref(ref: str) -> str | None:
    ref = ref.strip()
    if _CODE_REF_RE.match(ref):
        return "code"
    if _PR_REF_RE.match(ref):
        return "pr"
    if _ISSUE_REF_RE.match(ref):
        return "issue"
    if _COMMIT_REF_RE.match(ref):
        return "commit"
    if ref.lower().startswith(("memory ", "branch ", "ci @")):
        return "memory" if ref.lower().startswith("memory") else "commit"
    return None


def extract_citations(answer: str) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for m in _CITE_RE.finditer(_normalize_dashes(answer)):
        ref = m.group(1).strip()
        kind = classify_ref(ref)
        if kind and ref not in seen:
            seen.add(ref)
            citations.append(Citation(ref=ref, kind=kind))
    return citations


def _normalize_code_ref(ref: str) -> tuple[str, int, int] | None:
    m = re.match(r"^([\w./\\-]+):(\d+)(?:-(\d+))?$", ref)
    if not m:
        return None
    start = int(m.group(2))
    end = int(m.group(3)) if m.group(3) else start
    return m.group(1).replace("\\", "/"), start, end


def is_supported(ref: str, evidence_refs: list[str]) -> bool:
    """A citation is supported if it matches (or falls within) an evidence ref."""
    ref = _normalize_dashes(ref)
    evidence_refs = [_normalize_dashes(ev) for ev in evidence_refs]
    kind = classify_ref(ref)
    norm = ref.strip().lower().replace("  ", " ")
    for ev in evidence_refs:
        if ev.strip().lower() == norm:
            return True
    if kind == "code":
        cited = _normalize_code_ref(ref)
        if cited is None:
            return False
        c_path, c_start, c_end = cited
        for ev in evidence_refs:
            got = _normalize_code_ref(ev)
            if got is None:
                continue
            e_path, e_start, e_end = got
            # Allow partial paths cited by the model ("retriever.py" for
            # "retrieval/retriever.py"), matched at a path-component boundary.
            paths_match = e_path == c_path or e_path.endswith("/" + c_path)
            if paths_match and c_start >= e_start and c_end <= e_end:
                return True
    return False


def verify_citations(citations: list[Citation], evidence_refs: list[str]) -> dict:
    supported = [c for c in citations if is_supported(c.ref, evidence_refs)]
    unsupported = [c for c in citations if not is_supported(c.ref, evidence_refs)]
    return {
        "supported": [c.ref for c in supported],
        "unsupported": [c.ref for c in unsupported],
        "support_rate": len(supported) / len(citations) if citations else 1.0,
    }
