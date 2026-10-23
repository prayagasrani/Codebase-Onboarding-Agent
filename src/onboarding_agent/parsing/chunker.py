"""Semantic code chunking.

Symbols become chunks (functions, classes, methods). Code not covered by any
symbol — plus non-code files — is chunked with a line-based sliding window.
Every chunk carries file/line metadata for citations.
"""

from __future__ import annotations

import hashlib

from onboarding_agent.models import CodeChunk, Symbol

MAX_CHUNK_LINES = 120
WINDOW_LINES = 60
WINDOW_OVERLAP = 10


def _chunk_id(file_path: str, start: int, end: int, text: str) -> str:
    h = hashlib.sha1(f"{file_path}:{start}-{end}:{text[:200]}".encode("utf-8", "replace")).hexdigest()
    return h[:16]


def _make_chunk(file_path: str, language: str, lines: list[str], start: int, end: int,
                symbol: Symbol | None = None, imports: list[str] | None = None) -> CodeChunk:
    text = "\n".join(lines[start - 1:end])
    return CodeChunk(
        chunk_id=_chunk_id(file_path, start, end, text),
        file_path=file_path,
        start_line=start,
        end_line=end,
        text=text,
        language=language,
        symbol_name=symbol.qualified_name if symbol else None,
        symbol_kind=symbol.kind if symbol else None,
        imports=imports or [],
    )


class Chunker:
    def chunk_file(self, file_path: str, text: str, language: str,
                   symbols: list[Symbol], imports: list[str]) -> list[CodeChunk]:
        lines = text.splitlines()
        if not lines:
            return []
        chunks: list[CodeChunk] = []
        covered = [False] * (len(lines) + 1)

        # Prefer leaf-level symbols: skip classes whose methods are separate symbols
        # unless the class body is small.
        symbol_chunks = self._select_symbols(symbols)
        for sym in symbol_chunks:
            start, end = sym.start_line, min(sym.end_line, len(lines))
            if end < start:
                continue
            # Split oversized symbols into windows but keep the symbol label.
            for w_start in range(start, end + 1, MAX_CHUNK_LINES):
                w_end = min(w_start + MAX_CHUNK_LINES - 1, end)
                chunks.append(_make_chunk(file_path, language, lines, w_start, w_end, sym, imports))
            for i in range(start, end + 1):
                covered[i] = True

        # Sliding window over uncovered regions (module-level code, docs, configs).
        i = 1
        n = len(lines)
        while i <= n:
            if covered[i]:
                i += 1
                continue
            j = i
            while j <= n and not covered[j]:
                j += 1
            region_end = j - 1
            k = i
            while k <= region_end:
                w_end = min(k + WINDOW_LINES - 1, region_end)
                if any(line.strip() for line in lines[k - 1:w_end]):
                    chunks.append(_make_chunk(file_path, language, lines, k, w_end, None, imports))
                if w_end >= region_end:
                    break
                k = w_end + 1 - WINDOW_OVERLAP
            i = j

        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks

    @staticmethod
    def _select_symbols(symbols: list[Symbol]) -> list[Symbol]:
        """Choose which symbols become chunks: methods + functions + small classes."""
        selected: list[Symbol] = []
        classes = [s for s in symbols if s.kind == "class"]
        methods_by_parent: dict[str, list[Symbol]] = {}
        for s in symbols:
            if s.kind == "method" and s.parent:
                methods_by_parent.setdefault(s.parent, []).append(s)
        for s in symbols:
            if s.kind in ("function", "method"):
                selected.append(s)
            elif s.kind == "class":
                span = s.end_line - s.start_line + 1
                if s.name not in methods_by_parent or span <= MAX_CHUNK_LINES:
                    selected.append(s)
                else:
                    # class header (docstring + attributes) up to the first method
                    first_method = min(m.start_line for m in methods_by_parent[s.name])
                    if first_method > s.start_line:
                        header = Symbol(
                            name=s.name, kind="class", file_path=s.file_path,
                            start_line=s.start_line, end_line=first_method - 1,
                            parent=s.parent, signature=s.signature,
                        )
                        selected.append(header)
        # Deduplicate identical spans
        seen: set[tuple[int, int, str]] = set()
        unique: list[Symbol] = []
        for s in selected:
            key = (s.start_line, s.end_line, s.qualified_name)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        _ = classes  # kept for clarity; classes handled above
        return unique
