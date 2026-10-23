"""Code-aware tokenization shared by BM25 and the hash embedder."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Lowercased tokens with camelCase/snake_case sub-tokens included."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lower = raw.lower()
        tokens.append(lower)
        parts: list[str] = []
        for piece in raw.split("_"):
            if piece:
                parts.extend(_CAMEL_RE.split(piece))
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts if len(p) > 1)
    return tokens
