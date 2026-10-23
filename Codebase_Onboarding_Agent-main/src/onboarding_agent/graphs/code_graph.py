"""Lightweight dependency / call graph.

Nodes: files and symbols. Edges:
  - file --imports--> file        (resolved import statements)
  - symbol --calls--> symbol      (name-based call detection within chunk text)
  - file --defines--> symbol
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx

from onboarding_agent.models import CodeChunk, Symbol

_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_PY_KEYWORDS = {
    "if", "for", "while", "return", "print", "len", "range", "isinstance", "super",
    "str", "int", "float", "dict", "list", "set", "tuple", "type", "open", "enumerate",
    "zip", "sorted", "min", "max", "sum", "any", "all", "getattr", "setattr", "hasattr",
}


class CodeGraph:
    def __init__(self) -> None:
        self.g = nx.DiGraph()

    # ------------------------------------------------------------------ build
    def add_file(self, file_path: str, symbols: list[Symbol], imports: list[str],
                 all_files: set[str]) -> None:
        self.remove_file(file_path)
        self.g.add_node(f"file:{file_path}", kind="file")
        for sym in symbols:
            node = f"sym:{file_path}::{sym.qualified_name}"
            self.g.add_node(node, kind=sym.kind, name=sym.qualified_name,
                            file=file_path, start=sym.start_line, end=sym.end_line)
            self.g.add_edge(f"file:{file_path}", node, kind="defines")
        for target in self._resolve_imports(file_path, imports, all_files):
            self.g.add_node(f"file:{target}", kind="file")
            self.g.add_edge(f"file:{file_path}", f"file:{target}", kind="imports")

    def add_calls(self, chunks: list[CodeChunk], symbol_index: dict[str, list[str]]) -> None:
        """Detect call edges by matching called names against known symbol names."""
        for chunk in chunks:
            if not chunk.symbol_name:
                continue
            src = f"sym:{chunk.file_path}::{chunk.symbol_name}"
            if src not in self.g:
                continue
            called = {m.group(1) for m in _CALL_RE.finditer(chunk.text)} - _PY_KEYWORDS
            for name in called:
                for target in symbol_index.get(name, []):
                    if target != src and target in self.g:
                        self.g.add_edge(src, target, kind="calls")

    def remove_file(self, file_path: str) -> None:
        node = f"file:{file_path}"
        if node in self.g:
            symbols = [n for n in self.g.successors(node)
                       if self.g.edges[node, n].get("kind") == "defines"]
            self.g.remove_nodes_from(symbols)
            self.g.remove_node(node)

    @staticmethod
    def _resolve_imports(file_path: str, imports: list[str], all_files: set[str]) -> set[str]:
        """Best-effort resolution of import statements to repo-relative files."""
        resolved: set[str] = set()
        modules: list[str] = []
        for imp in imports:
            imp = imp.strip()
            if m := re.match(r"from\s+([\w.]+)\s+import", imp):
                modules.append(m.group(1))
            elif m := re.match(r"import\s+(.+)", imp):
                for part in m.group(1).split(","):
                    modules.append(part.strip().split(" as ")[0].split(".")[0])
            elif m := re.search(r"from\s+['\"](.+?)['\"]", imp):
                modules.append(m.group(1))
        for mod in modules:
            candidate_paths = {
                mod.replace(".", "/") + ".py",
                mod.replace(".", "/") + "/__init__.py",
                mod.lstrip("./") + ".py",
                mod.lstrip("./") + ".ts",
                mod.lstrip("./") + ".js",
            }
            for f in all_files:
                if f in candidate_paths or f.endswith("/" + mod.split(".")[-1] + ".py"):
                    resolved.add(f)
        return resolved

    # ------------------------------------------------------------------ query
    def trace_dependencies(self, target: str, direction: str = "both", depth: int = 2) -> dict:
        """Trace what `target` (file path or symbol name) depends on / is used by."""
        nodes = self._find_nodes(target)
        if not nodes:
            return {"target": target, "found": False, "dependencies": [], "dependents": []}
        deps: set[str] = set()
        dependents: set[str] = set()
        for node in nodes:
            if direction in ("both", "downstream"):
                deps |= set(nx.dfs_preorder_nodes(self.g, node, depth_limit=depth)) - {node}
            if direction in ("both", "upstream"):
                dependents |= set(
                    nx.dfs_preorder_nodes(self.g.reverse(copy=False), node, depth_limit=depth)
                ) - {node}
        return {
            "target": target,
            "found": True,
            "dependencies": sorted(self._label(n) for n in deps),
            "dependents": sorted(self._label(n) for n in dependents),
        }

    def file_fan_in(self) -> list[tuple[str, int]]:
        """Files ranked by import fan-in — a proxy for 'major components'."""
        counts = []
        for node, data in self.g.nodes(data=True):
            if data.get("kind") == "file":
                fan_in = sum(
                    1 for pred in self.g.predecessors(node)
                    if self.g.edges[pred, node].get("kind") == "imports"
                )
                counts.append((node.removeprefix("file:"), fan_in))
        return sorted(counts, key=lambda kv: -kv[1])

    def _find_nodes(self, target: str) -> list[str]:
        hits = []
        for node, data in self.g.nodes(data=True):
            if node == f"file:{target}":
                hits.append(node)
            elif data.get("kind") != "file" and data.get("name") in (target, target.split(".")[-1]):
                hits.append(node)
            elif data.get("kind") == "file" and node.removeprefix("file:").endswith(target):
                hits.append(node)
        return hits

    def _label(self, node: str) -> str:
        data = self.g.nodes[node]
        if data.get("kind") == "file":
            return node.removeprefix("file:")
        return f"{data.get('file')}::{data.get('name')} ({data.get('kind')})"

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = nx.node_link_data(self.g, edges="links")
        (directory / "code_graph.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "CodeGraph":
        graph = cls()
        path = directory / "code_graph.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            graph.g = nx.node_link_graph(payload, directed=True, edges="links")
        return graph
