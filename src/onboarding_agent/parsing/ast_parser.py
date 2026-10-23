"""AST-aware parsing.

Uses Tree-sitter (via ``tree-sitter-language-pack``) when installed for precise
symbol extraction; falls back to a regex/indentation parser otherwise, so the
pipeline stays runnable without native wheels.
"""

from __future__ import annotations

import re

from onboarding_agent.logging import get_logger
from onboarding_agent.models import Symbol

log = get_logger(__name__)

# Tree-sitter node types per language → symbol kind
_TS_NODE_KINDS: dict[str, dict[str, str]] = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
    },
    "typescript": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "interface_declaration": "class",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "class",
    },
    "java": {"method_declaration": "method", "class_declaration": "class", "interface_declaration": "class"},
    "rust": {"function_item": "function", "struct_item": "class", "impl_item": "class", "trait_item": "class"},
    "ruby": {"method": "function", "class": "class", "module": "class"},
}

_TS_IMPORT_NODES = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "go": {"import_declaration"},
    "java": {"import_declaration"},
    "rust": {"use_declaration"},
    "ruby": set(),
}

_PARSER_CACHE: dict[str, object] = {}


def _get_ts_parser(language: str):
    """Return a cached tree-sitter parser for the language, or None."""
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]
    parser = None
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore

        parser = get_parser(language)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - any failure means "unavailable"
        parser = None
    _PARSER_CACHE[language] = parser
    return parser


class AstParser:
    """Extracts symbols (functions/classes/methods) and imports from source."""

    def parse(self, text: str, file_path: str, language: str) -> tuple[list[Symbol], list[str]]:
        if language in _TS_NODE_KINDS:
            parser = _get_ts_parser(language)
            if parser is not None:
                try:
                    return self._parse_tree_sitter(parser, text, file_path, language)
                except Exception as exc:  # noqa: BLE001
                    log.warning("tree_sitter_parse_failed", file=file_path, error=str(exc))
        return self._parse_regex(text, file_path, language)

    # ------------------------------------------------------------------ tree-sitter
    def _parse_tree_sitter(
        self, parser, text: str, file_path: str, language: str
    ) -> tuple[list[Symbol], list[str]]:
        data = text.encode("utf-8", errors="replace")
        tree = parser.parse(data)
        kinds = _TS_NODE_KINDS[language]
        import_nodes = _TS_IMPORT_NODES.get(language, set())
        symbols: list[Symbol] = []
        imports: list[str] = []

        def node_name(node) -> str | None:
            named = node.child_by_field_name("name")
            if named is not None:
                return data[named.start_byte:named.end_byte].decode("utf-8", errors="replace")
            for child in node.children:
                if child.type in ("identifier", "type_identifier", "field_identifier", "constant"):
                    return data[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            return None

        def walk(node, parent_class: str | None) -> None:
            for child in node.children:
                ctype = child.type
                if ctype in import_nodes:
                    imports.append(
                        data[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                    )
                if ctype in kinds:
                    name = node_name(child)
                    if name:
                        kind = kinds[ctype]
                        if kind == "function" and parent_class:
                            kind = "method"
                        first_line = data[child.start_byte:child.end_byte].split(b"\n", 1)[0]
                        symbols.append(
                            Symbol(
                                name=name,
                                kind=kind,
                                file_path=file_path,
                                start_line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                parent=parent_class,
                                signature=first_line.decode("utf-8", errors="replace").strip(),
                            )
                        )
                        walk(child, name if kind == "class" else parent_class)
                        continue
                walk(child, parent_class)

        walk(tree.root_node, None)
        return symbols, imports

    # ------------------------------------------------------------------ regex fallback
    _PY_DEF = re.compile(r"^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
    _PY_CLASS = re.compile(r"^(\s*)class\s+([A-Za-z_]\w*)\b")
    _PY_IMPORT = re.compile(r"^\s*(?:from\s+[\w.]+\s+import\s+.+|import\s+[\w., ]+)")
    _JS_FUNC = re.compile(
        r"^(\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"
    )
    _JS_CLASS = re.compile(r"^(\s*)(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)")
    _JS_ARROW = re.compile(
        r"^(\s*)(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(?[\w\s,{}:$]*\)?\s*=>"
    )
    _JS_IMPORT = re.compile(r"^\s*import\s+.+?from\s+['\"].+?['\"]|^\s*(?:const|let)\s+.+?=\s*require\(")
    _GO_FUNC = re.compile(r"^func\s+(?:\([^)]*\)\s+)?([A-Za-z_]\w*)\s*\(")
    _GO_TYPE = re.compile(r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)\b")

    def _parse_regex(self, text: str, file_path: str, language: str) -> tuple[list[Symbol], list[str]]:
        lines = text.splitlines()
        symbols: list[Symbol] = []
        imports: list[str] = []

        if language == "python":
            self._regex_python(lines, file_path, symbols, imports)
        elif language in ("javascript", "typescript"):
            self._regex_js(lines, file_path, symbols, imports)
        elif language == "go":
            for i, line in enumerate(lines):
                if m := self._GO_FUNC.match(line):
                    end = self._brace_block_end(lines, i)
                    symbols.append(Symbol(m.group(1), "function", file_path, i + 1, end + 1,
                                          signature=line.strip()))
                elif m := self._GO_TYPE.match(line):
                    end = self._brace_block_end(lines, i)
                    symbols.append(Symbol(m.group(1), "class", file_path, i + 1, end + 1,
                                          signature=line.strip()))
                elif line.startswith("import"):
                    imports.append(line.strip())
        return symbols, imports

    def _regex_python(self, lines: list[str], file_path: str,
                      symbols: list[Symbol], imports: list[str]) -> None:
        class_stack: list[tuple[int, str]] = []  # (indent, name)
        for i, line in enumerate(lines):
            if self._PY_IMPORT.match(line):
                imports.append(line.strip())
                continue
            m_class = self._PY_CLASS.match(line)
            m_def = self._PY_DEF.match(line)
            m = m_class or m_def
            if not m:
                continue
            indent = len(m.group(1))
            name = m.group(2)
            while class_stack and class_stack[-1][0] >= indent:
                class_stack.pop()
            parent = class_stack[-1][1] if class_stack else None
            end = self._indent_block_end(lines, i, indent)
            if m_class:
                symbols.append(Symbol(name, "class", file_path, i + 1, end + 1,
                                      parent=parent, signature=line.strip()))
                class_stack.append((indent, name))
            else:
                kind = "method" if parent else "function"
                symbols.append(Symbol(name, kind, file_path, i + 1, end + 1,
                                      parent=parent, signature=line.strip()))

    def _regex_js(self, lines: list[str], file_path: str,
                  symbols: list[Symbol], imports: list[str]) -> None:
        for i, line in enumerate(lines):
            if self._JS_IMPORT.match(line):
                imports.append(line.strip())
                continue
            for pattern, kind in ((self._JS_CLASS, "class"), (self._JS_FUNC, "function"),
                                  (self._JS_ARROW, "function")):
                if m := pattern.match(line):
                    end = self._brace_block_end(lines, i)
                    symbols.append(Symbol(m.group(2), kind, file_path, i + 1, end + 1,
                                          signature=line.strip()))
                    break

    @staticmethod
    def _indent_block_end(lines: list[str], start: int, indent: int) -> int:
        """End line of an indentation-delimited block (Python)."""
        end = start
        for j in range(start + 1, len(lines)):
            stripped = lines[j].strip()
            if not stripped:
                continue
            if len(lines[j]) - len(lines[j].lstrip()) <= indent and not stripped.startswith(("#",)):
                break
            end = j
        return end

    @staticmethod
    def _brace_block_end(lines: list[str], start: int) -> int:
        """End line of a brace-delimited block (JS/TS/Go)."""
        depth = 0
        opened = False
        for j in range(start, min(len(lines), start + 800)):
            depth += lines[j].count("{") - lines[j].count("}")
            if "{" in lines[j]:
                opened = True
            if opened and depth <= 0:
                return j
        return start
