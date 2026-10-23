"""Agent tools: schemas (name / description / JSON ``input_schema``) + executor.

``GroqLLM`` translates these to OpenAI function-calling format on the wire.

The router selects which tool group(s) the agent sees, so a pure code
question never hits the GitHub API and vice versa.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from onboarding_agent.agent.llm import ToolOutcome
from onboarding_agent.logging import get_logger

log = get_logger(__name__)

CODE_TOOLS: list[dict] = [
    {
        "name": "search_code",
        "description": (
            "Hybrid semantic + lexical search over the indexed codebase. Call this first for any "
            "question about how or where something is implemented. Returns chunks with "
            "file:line citations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language or code query"},
                "top_k": {"type": "integer", "description": "Number of results (default 8)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_file",
        "description": "Read a file (or a line range) from the indexed repository, verbatim.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative path"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_symbol",
        "description": "Look up a function/class/method by name; returns definition location and source.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Symbol name, e.g. 'LoginService' or 'authenticate'"}},
            "required": ["name"],
        },
    },
    {
        "name": "trace_dependencies",
        "description": (
            "Trace the dependency/call graph around a file or symbol: what it depends on "
            "(downstream) and what uses it (upstream)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "File path or symbol name"},
                "direction": {"type": "string", "enum": ["both", "upstream", "downstream"]},
            },
            "required": ["target"],
        },
    },
]

GITHUB_TOOLS: list[dict] = [
    {
        "name": "search_github",
        "description": "Search this repo's GitHub issues/PRs, commits, or code for a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "kind": {"type": "string", "enum": ["issues", "commits", "code"]},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_pr",
        "description": "Fetch one pull request: title, state, body, changed files.",
        "input_schema": {
            "type": "object",
            "properties": {"number": {"type": "integer"}},
            "required": ["number"],
        },
    },
    {
        "name": "get_commit",
        "description": "Fetch one commit: message, author, changed files.",
        "input_schema": {
            "type": "object",
            "properties": {"sha": {"type": "string"}},
            "required": ["sha"],
        },
    },
    {
        "name": "list_pull_requests",
        "description": "List recent pull requests (default: open, most recently updated).",
        "input_schema": {
            "type": "object",
            "properties": {"state": {"type": "string", "enum": ["open", "closed", "all"]},
                           "limit": {"type": "integer"}},
        },
    },
    {
        "name": "list_issues",
        "description": "List recent issues (excludes PRs).",
        "input_schema": {
            "type": "object",
            "properties": {"state": {"type": "string", "enum": ["open", "closed", "all"]},
                           "limit": {"type": "integer"}},
        },
    },
    {
        "name": "get_ci_status",
        "description": "CI/check-run status for a branch or commit (default: default branch head).",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string", "description": "Branch, tag, or SHA"}},
        },
    },
    {
        "name": "get_file_history",
        "description": "Recent commits that touched a specific file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
]

MEMORY_TOOLS: list[dict] = [
    {
        "name": "search_memory",
        "description": (
            "Search this user's past questions and explanations. Use to avoid repeating "
            "explanations and to answer 'what did I ask before' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def tools_for_route(route: str, github_available: bool) -> list[dict]:
    groups = {
        "code": CODE_TOOLS + MEMORY_TOOLS,
        "memory": MEMORY_TOOLS + CODE_TOOLS,
        "github": (GITHUB_TOOLS if github_available else []) + CODE_TOOLS[:1],
        "mixed": CODE_TOOLS + (GITHUB_TOOLS if github_available else []) + MEMORY_TOOLS,
    }
    return groups.get(route, CODE_TOOLS)


class ToolExecutor:
    """Dispatches tool calls to the index, retriever, GitHub client, and memory."""

    def __init__(self, index, retriever, memory, user_id: str, github=None,
                 retrieval_mode: str = "hybrid_rrf"):
        self.index = index
        self.retriever = retriever
        self.memory = memory
        self.github = github
        self.user_id = user_id
        self.retrieval_mode = retrieval_mode

    def __call__(self, name: str, args: dict) -> ToolOutcome:
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return ToolOutcome(f"Unknown tool: {name}", is_error=True)
            # Drop hallucinated extra arguments instead of failing the call.
            accepted = inspect.signature(handler).parameters
            known = {k: v for k, v in args.items() if k in accepted}
            if len(known) < len(args):
                log.warning("tool_extra_args_dropped", tool=name,
                            dropped=sorted(set(args) - set(known)))
            return handler(**known)
        except Exception as exc:  # noqa: BLE001
            log.warning("tool_failed", tool=name, error=str(exc))
            return ToolOutcome(f"Tool {name} failed: {exc}", is_error=True)

    # ------------------------------------------------------------------ code tools
    def _tool_search_code(self, query: str, top_k: int = 8) -> ToolOutcome:
        results = self.retriever.search(query, top_k=top_k, mode=self.retrieval_mode)
        payload = [
            {
                "ref": r.chunk.citation,
                "symbol": r.chunk.symbol_name,
                "language": r.chunk.language,
                "text": r.chunk.text[:900],
            }
            for r in results
        ]
        return ToolOutcome(payload, refs=[r.chunk.citation for r in results])

    def _tool_get_file(self, path: str, start_line: int | None = None,
                       end_line: int | None = None) -> ToolOutcome:
        root = Path(self.index.manifest.get("repo_root", "."))
        target = (root / path).resolve()
        if not str(target).startswith(str(root.resolve())):
            return ToolOutcome("Path escapes repository root", is_error=True)
        if not target.exists():
            known = [p for p in self.index.manifest["files"] if path.lower() in p.lower()][:5]
            return ToolOutcome(f"File not found: {path}. Similar: {known}", is_error=True)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line or 1)
        end = min(len(lines), end_line or len(lines))
        text = "\n".join(lines[start - 1:end])
        ref = f"{path}:{start}-{end}"
        return ToolOutcome({"ref": ref, "text": text[:8000]}, refs=[ref])

    def _tool_get_symbol(self, name: str) -> ToolOutcome:
        matches = []
        wanted = name.split(".")[-1].lower()
        for _path, syms in self.index.symbols.items():
            for s in syms:
                if s.name.lower() == wanted or s.qualified_name.lower() == name.lower():
                    matches.append(s)
        if not matches:
            return ToolOutcome(f"Symbol not found: {name}", is_error=True)
        payload, refs = [], []
        for s in matches[:5]:
            ref = f"{s.file_path}:{s.start_line}-{s.end_line}"
            refs.append(ref)
            source = ""
            for chunk in self.index.chunks.values():
                if (chunk.file_path == s.file_path
                        and chunk.start_line <= s.start_line <= chunk.end_line):
                    source = chunk.text[:2000]
                    break
            payload.append({"ref": ref, "kind": s.kind, "qualified_name": s.qualified_name,
                            "signature": s.signature, "source": source})
        return ToolOutcome(payload, refs=refs)

    def _tool_trace_dependencies(self, target: str, direction: str = "both") -> ToolOutcome:
        result = self.index.graph.trace_dependencies(target, direction=direction)
        refs = []
        if result.get("found"):
            refs = [target]
        return ToolOutcome(result, refs=refs)

    # ------------------------------------------------------------------ github tools
    def _require_github(self):
        if self.github is None:
            raise RuntimeError("GitHub is not configured (set GITHUB_TOKEN / GITHUB_REPO)")
        return self.github

    def _tool_search_github(self, query: str, kind: str = "issues") -> ToolOutcome:
        items = self._require_github().search(query, kind=kind)
        return ToolOutcome(items, refs=[i["ref"] for i in items])

    def _tool_get_pr(self, number: int) -> ToolOutcome:
        pr = self._require_github().pull_request(number)
        return ToolOutcome(pr, refs=[pr["ref"]])

    def _tool_get_commit(self, sha: str) -> ToolOutcome:
        c = self._require_github().commit(sha)
        return ToolOutcome(c, refs=[c["ref"]])

    def _tool_list_pull_requests(self, state: str = "open", limit: int = 10) -> ToolOutcome:
        prs = self._require_github().pull_requests(state=state, limit=limit)
        return ToolOutcome(prs, refs=[p["ref"] for p in prs])

    def _tool_list_issues(self, state: str = "open", limit: int = 10) -> ToolOutcome:
        items = self._require_github().issues(state=state, limit=limit)
        return ToolOutcome(items, refs=[i["ref"] for i in items])

    def _tool_get_ci_status(self, ref: str = "HEAD") -> ToolOutcome:
        status = self._require_github().ci_status(ref)
        return ToolOutcome(status, refs=[status["ref"]])

    def _tool_get_file_history(self, path: str, limit: int = 10) -> ToolOutcome:
        commits = self._require_github().file_history(path, limit=limit)
        return ToolOutcome(commits, refs=[c["ref"] for c in commits])

    # ------------------------------------------------------------------ memory tools
    def _tool_search_memory(self, query: str) -> ToolOutcome:
        hits = self.memory.search(self.user_id, query)
        return ToolOutcome(hits, refs=[f"memory {h['asked_at'][:10]}" for h in hits])
