"""LangGraph agent: route -> gather -> synthesize -> memorize.

Two execution paths share the same graph shape:
- LLM path: a Groq tool-use loop gathers evidence and writes the answer.
- Offline path (no API key): deterministic tool calls gather evidence and an
  extractive composer produces a cited answer. Keeps dev/test/eval hermetic.
"""

from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from onboarding_agent.agent.citations import extract_citations, verify_citations
from onboarding_agent.agent.llm import LLM, ToolOutcome
from onboarding_agent.agent.prompts import AGENT_SYSTEM, INSUFFICIENT_EVIDENCE, MEMORY_BLOCK
from onboarding_agent.agent.router import QueryRouter
from onboarding_agent.agent.tools import ToolExecutor, tools_for_route
from onboarding_agent.logging import get_logger

log = get_logger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    user_id: str
    route: str
    memory_context: dict
    evidence: list[dict]
    answer: str
    citations: list[dict]
    verification: dict
    latency_s: float
    _t0: float


class OnboardingAgent:
    def __init__(
        self,
        executor: ToolExecutor,
        llm: LLM | None = None,
        github_available: bool = False,
        update_memory: bool = True,
    ):
        self.executor = executor
        self.llm = llm
        self.router = QueryRouter(llm)
        self.github_available = github_available
        self.update_memory = update_memory
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("route", self._node_route)
        g.add_node("gather", self._node_gather)
        g.add_node("synthesize", self._node_synthesize)
        g.add_node("memorize", self._node_memorize)
        g.set_entry_point("route")
        g.add_edge("route", "gather")
        g.add_edge("gather", "synthesize")
        g.add_edge("synthesize", "memorize")
        g.add_edge("memorize", END)
        return g.compile()

    def ask(self, question: str, user_id: str = "default") -> dict[str, Any]:
        state: AgentState = {"question": question, "user_id": user_id, "_t0": time.time()}
        final = self.graph.invoke(state)
        return {
            "answer": final.get("answer", ""),
            "citations": final.get("citations", []),
            "verification": final.get("verification", {}),
            "route": final.get("route", ""),
            "evidence": final.get("evidence", []),
            "latency_s": final.get("latency_s", 0.0),
        }

    # ------------------------------------------------------------------ nodes
    def _node_route(self, state: AgentState) -> AgentState:
        route = self.router.route(state["question"])
        memory_context = self.executor.memory.familiarity_summary(
            state["user_id"], state["question"]
        )
        log.info("routed", route=route, question=state["question"][:80])
        return {"route": route, "memory_context": memory_context}

    def _node_gather(self, state: AgentState) -> AgentState:
        if self.llm is not None:
            return {}  # LLM path gathers inside the tool loop in synthesize
        return {"evidence": self._gather_deterministic(state)}

    def _node_synthesize(self, state: AgentState) -> AgentState:
        if self.llm is not None:
            answer, evidence = self._synthesize_llm(state)
        else:
            evidence = state.get("evidence", [])
            answer = self._compose_extractive(state, evidence)
        citations = extract_citations(answer)
        evidence_refs = [ref for e in evidence for ref in e.get("refs", [])]
        verification = verify_citations(citations, evidence_refs)
        return {
            "answer": answer,
            "evidence": evidence,
            "citations": [c.__dict__ for c in citations],
            "verification": verification,
            "latency_s": round(time.time() - state["_t0"], 3),
        }

    def _node_memorize(self, state: AgentState) -> AgentState:
        if not self.update_memory or not state.get("answer"):
            return {}
        files = sorted({
            ref.split(":")[0] for c in state.get("citations", [])
            for ref in [c["ref"]] if c["kind"] == "code"
        })
        self.executor.memory.record_interaction(
            user_id=state["user_id"],
            question=state["question"],
            answer=state["answer"],
            citations=[c["ref"] for c in state.get("citations", [])],
            files=files,
        )
        return {}

    # ------------------------------------------------------------------ LLM path
    def _synthesize_llm(self, state: AgentState) -> tuple[str, list[dict]]:
        mc = state.get("memory_context", {})
        memory_block = ""
        if mc.get("previously_discussed") or mc.get("familiar_concepts"):
            history = "; ".join(
                f"Q: {p['question'][:100]}" for p in mc.get("previously_discussed", [])
            ) or "none"
            memory_block = MEMORY_BLOCK.format(
                familiar=", ".join(mc.get("familiar_concepts", [])) or "none",
                new=", ".join(mc.get("new_concepts", [])) or "none",
                history=history,
            )
        system = AGENT_SYSTEM.format(memory_block=memory_block)
        tools = tools_for_route(state["route"], self.github_available)
        answer, evidence = self.llm.run_tools(
            system=system,
            user_message=state["question"],
            tools=tools,
            execute=self.executor,
        )
        if not answer.strip():
            answer = INSUFFICIENT_EVIDENCE.format(
                github_part=" or GitHub" if self.github_available else "",
                missing="a grounded answer from the agent loop",
            )
        return answer, evidence

    # ------------------------------------------------------------------ offline path
    def _gather_deterministic(self, state: AgentState) -> list[dict]:
        q = state["question"]
        route = state["route"]
        evidence: list[dict] = []

        def run(tool: str, **args) -> ToolOutcome:
            outcome = self.executor(tool, args)
            evidence.append({
                "tool": tool, "input": args, "refs": outcome.refs,
                "result_snippet": outcome.as_text(1200),
                "payload": outcome.payload if not outcome.is_error else None,
                "is_error": outcome.is_error,
            })
            return outcome

        if route in ("code", "mixed", "memory"):
            run("search_code", query=q, top_k=8)
            # Symbol lookup for CamelCase / snake_case identifiers in the question.
            for ident in re.findall(r"\b[A-Z][A-Za-z0-9]+\b|\b[a-z]+_[a-z_]+\b", q)[:2]:
                run("get_symbol", name=ident)
            if re.search(r"\bdepend|\buses\b|\bused by\b|\bcall(s|ed)?\b", q.lower()):
                tokens = re.findall(r"[\w./]+\.(?:py|ts|js|go)|\b[A-Z][A-Za-z0-9]+\b", q)
                if tokens:
                    run("trace_dependencies", target=tokens[0])
        if route in ("github", "mixed") and self.github_available:
            run("list_pull_requests", state="open", limit=10)
            run("search_github", query=q, kind="issues")
            if re.search(r"\bcommit|\bchange[sd]?\b|\brecent", q.lower()):
                run("search_github", query=q, kind="commits")
            if re.search(r"\bci\b|\bbuild\b|\bpipeline", q.lower()):
                run("get_ci_status")
        if route == "memory":
            run("search_memory", query=q)
        return evidence

    def _compose_extractive(self, state: AgentState, evidence: list[dict]) -> str:
        """Cited, extractive answer from gathered evidence (no LLM)."""
        parts: list[str] = []
        code_hits = []
        for e in evidence:
            if e["tool"] == "search_code" and e.get("payload"):
                code_hits = e["payload"][:5]
                break
        if code_hits:
            parts.append("Most relevant code locations:")
            for hit in code_hits:
                label = hit.get("symbol") or "module-level code"
                snippet = " ".join(hit["text"].split())[:220]
                parts.append(f"- {label} [{hit['ref']}]: {snippet}")

        for e in evidence:
            if e["tool"] == "get_symbol" and e.get("payload"):
                for s in e["payload"][:2]:
                    parts.append(
                        f"- Definition of {s['qualified_name']} ({s['kind']}) at [{s['ref']}]: "
                        f"{s['signature']}"
                    )
            elif e["tool"] == "trace_dependencies" and e.get("payload", {}) \
                    and e["payload"].get("found"):
                p = e["payload"]
                parts.append(
                    f"- {p['target']} depends on: {', '.join(p['dependencies'][:8]) or 'nothing indexed'}; "
                    f"used by: {', '.join(p['dependents'][:8]) or 'nothing indexed'}"
                )
            elif e["tool"] == "list_pull_requests" and e.get("payload"):
                parts.append("Open pull requests (current development activity):")
                for pr in e["payload"][:5]:
                    parts.append(f"- [{pr['ref']}] {pr['title']} ({pr['state']}, by {pr['author']})")
            elif e["tool"] == "search_github" and e.get("payload"):
                for item in e["payload"][:5]:
                    title = item.get("title") or item.get("message", "")
                    parts.append(f"- [{item['ref']}] {title}")
            elif e["tool"] == "get_ci_status" and e.get("payload"):
                p = e["payload"]
                parts.append(f"- CI status [{p['ref']}]: {p['overall']}")
            elif e["tool"] == "search_memory" and e.get("payload"):
                parts.append("From your previous conversations:")
                for h in e["payload"][:3]:
                    parts.append(f"- ({h['asked_at'][:10]}) Q: {h['question'][:120]}")

        if not parts:
            return INSUFFICIENT_EVIDENCE.format(
                github_part=" or GitHub" if self.github_available else "",
                missing="any matching code chunks, symbols, or development activity",
            )
        header = (
            "Evidence found for your question (extractive mode — set GROQ_API_KEY "
            "for synthesized answers):"
        )
        return header + "\n" + "\n".join(parts)
