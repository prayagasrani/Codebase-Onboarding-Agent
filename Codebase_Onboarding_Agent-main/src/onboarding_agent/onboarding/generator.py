"""'Onboard Me' — a structured project walkthrough for new developers.

Gathers deterministic facts from the index, graph, and GitHub, then (when an
LLM is configured) synthesizes a narrative guide; otherwise emits the
structured facts directly. Every section carries citations.
"""

from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath

from onboarding_agent.logging import get_logger

log = get_logger(__name__)

SECTION_QUERIES = {
    "authentication_and_data_flow": "authentication login session token data flow request handling",
    "testing": "tests test fixtures pytest unittest coverage",
    "deployment": "deployment docker dockerfile ci cd release build pipeline",
    "configuration": "configuration settings environment variables config",
}

ENTRYPOINT_HINTS = (
    "main.py", "app.py", "cli.py", "manage.py", "index.ts", "index.js", "main.go",
    "server.py", "wsgi.py", "asgi.py", "__main__.py", "main.ts", "main.js",
)

DOC_FILES = ("readme.md", "readme.rst", "contributing.md", "architecture.md", "docs/index.md")

SYNTHESIS_PROMPT = """\
You are writing an onboarding guide for a developer joining this repository.
Using ONLY the structured facts below (do not invent anything), write a concise
guide with these sections: Project Overview, Architecture, Entry Points,
Major Components, Important Dependencies, Authentication & Data Flow, Testing,
Deployment, Current Development Activity.

Cite sources inline with the refs given, e.g. [src/app.py:1-40] or [PR #12].
If a section has no facts, write "Not enough indexed evidence." for it.

FACTS:
{facts}
"""


class OnboardingGenerator:
    def __init__(self, service):  # service: onboarding_agent.service.AppService
        self.service = service

    # ------------------------------------------------------------------ facts
    def collect_facts(self) -> dict:
        index = self.service.index
        retriever = self.service.retriever
        facts: dict = {"repo_root": index.manifest.get("repo_root", ""), "stats": index.stats()}

        files = list(index.manifest.get("files", {}).keys())
        lang_counter = Counter(chunk.language for chunk in index.chunks.values())
        facts["languages"] = lang_counter.most_common(6)

        # Directory layout (top-level dirs by file count)
        dir_counter = Counter(
            (PurePosixPath(f).parts[0] if len(PurePosixPath(f).parts) > 1 else ".")
            for f in files
        )
        facts["top_level_layout"] = dir_counter.most_common(12)

        # Docs / overview
        facts["docs"] = []
        for f in files:
            if f.lower() in DOC_FILES or f.lower().endswith("readme.md"):
                head = self._file_head(f, 40)
                if head:
                    facts["docs"].append({"ref": f"{f}:1-40", "text": head})

        # Entry points
        facts["entry_points"] = []
        for f in files:
            name = PurePosixPath(f).name.lower()
            if name in ENTRYPOINT_HINTS or "if __name__" in self._file_head(f, 5, tail=True):
                facts["entry_points"].append(f)
        facts["entry_points"] = facts["entry_points"][:10]

        # Major components = files with highest import fan-in + biggest symbol counts
        fan_in = self.service.index.graph.file_fan_in()[:10]
        facts["major_components"] = [
            {"file": path, "imported_by": count} for path, count in fan_in if count > 0
        ]
        symbol_counts = sorted(
            ((path, len(syms)) for path, syms in index.symbols.items() if syms),
            key=lambda kv: -kv[1],
        )[:8]
        facts["symbol_rich_files"] = [{"file": p, "symbols": c} for p, c in symbol_counts]

        # Dependencies (manifest files)
        facts["dependency_manifests"] = [
            {"ref": f"{f}:1-40", "text": self._file_head(f, 40)}
            for f in files
            if PurePosixPath(f).name.lower() in
            ("pyproject.toml", "requirements.txt", "package.json", "go.mod", "cargo.toml",
             "gemfile", "pom.xml", "build.gradle")
        ][:4]

        # Topic sections via retrieval
        facts["sections"] = {}
        if index.is_built:
            for section, query in SECTION_QUERIES.items():
                hits = retriever.search(query, top_k=4, mode="hybrid_rrf")
                facts["sections"][section] = [
                    {"ref": r.chunk.citation, "symbol": r.chunk.symbol_name,
                     "snippet": " ".join(r.chunk.text.split())[:200]}
                    for r in hits
                ]

        # Current development activity
        facts["development_activity"] = {}
        gh = self.service.github
        if gh is not None:
            try:
                facts["development_activity"] = {
                    "open_prs": gh.pull_requests(state="open", limit=5),
                    "recent_commits": gh.commits(limit=5),
                    "open_issues": gh.issues(state="open", limit=5),
                    "ci": gh.ci_status(),
                }
            except Exception as exc:  # noqa: BLE001
                log.warning("onboard_github_failed", error=str(exc))
                facts["development_activity"] = {"error": str(exc)}
        return facts

    def _file_head(self, rel_path: str, n_lines: int, tail: bool = False) -> str:
        from pathlib import Path

        root = Path(self.service.index.manifest.get("repo_root", "."))
        p = root / rel_path
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        picked = lines[-n_lines:] if tail else lines[:n_lines]
        return "\n".join(picked)

    # ------------------------------------------------------------------ output
    def generate(self) -> dict:
        facts = self.collect_facts()
        if self.service.llm is not None:
            import json

            guide = self.service.llm.complete(
                SYNTHESIS_PROMPT.format(facts=json.dumps(facts, default=str)[:60000]),
                max_tokens=4096,
            )
            if guide.strip():
                return {"guide": guide, "facts": facts, "mode": "llm"}
        return {"guide": self._render_template(facts), "facts": facts, "mode": "template"}

    @staticmethod
    def _render_template(facts: dict) -> str:
        lines: list[str] = ["# Onboarding Guide", ""]
        lines.append("## Project Overview")
        lines.append(f"Repository: {facts['repo_root']}")
        lines.append(
            f"Indexed: {facts['stats']['files']} files, {facts['stats']['chunks']} chunks. "
            f"Languages: {', '.join(f'{lang} ({n})' for lang, n in facts['languages'])}."
        )
        for doc in facts["docs"][:1]:
            lines.append(f"From the README [{doc['ref']}]:")
            lines.append("> " + " ".join(doc["text"].split())[:400])
        lines.append("")
        lines.append("## Architecture (top-level layout)")
        for d, n in facts["top_level_layout"]:
            lines.append(f"- `{d}/` — {n} files")
        lines.append("")
        lines.append("## Entry Points")
        if facts["entry_points"]:
            lines.extend(f"- `{f}`" for f in facts["entry_points"])
        else:
            lines.append("Not enough indexed evidence.")
        lines.append("")
        lines.append("## Major Components")
        for c in facts["major_components"] or []:
            lines.append(f"- `{c['file']}` (imported by {c['imported_by']} files)")
        for c in facts["symbol_rich_files"][:5]:
            lines.append(f"- `{c['file']}` ({c['symbols']} symbols)")
        lines.append("")
        lines.append("## Important Dependencies")
        for dep in facts["dependency_manifests"]:
            lines.append(f"From [{dep['ref']}]:")
            lines.append("```\n" + dep["text"][:600] + "\n```")
        lines.append("")
        titles = {
            "authentication_and_data_flow": "## Authentication & Data Flow",
            "testing": "## Testing",
            "deployment": "## Deployment",
            "configuration": "## Configuration",
        }
        for key, title in titles.items():
            lines.append(title)
            hits = facts["sections"].get(key, [])
            if hits:
                for h in hits:
                    label = h["symbol"] or "module"
                    lines.append(f"- {label} [{h['ref']}]: {h['snippet']}")
            else:
                lines.append("Not enough indexed evidence.")
            lines.append("")
        lines.append("## Current Development Activity")
        activity = facts["development_activity"]
        if activity and "error" not in activity:
            for pr in activity.get("open_prs", []):
                lines.append(f"- Open [{pr['ref']}]: {pr['title']}")
            for c in activity.get("recent_commits", []):
                lines.append(f"- Recent [{c['ref']}]: {c['message']}")
            ci = activity.get("ci")
            if ci:
                lines.append(f"- CI [{ci['ref']}]: {ci['overall']}")
        else:
            lines.append("GitHub not configured — set GITHUB_TOKEN and GITHUB_REPO for live activity.")
        return "\n".join(lines)
