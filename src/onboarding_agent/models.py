"""Shared domain models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Symbol:
    """A named code entity extracted from the AST."""

    name: str
    kind: str  # "function" | "class" | "method" | "import"
    file_path: str
    start_line: int
    end_line: int
    parent: str | None = None  # enclosing class for methods
    signature: str = ""

    @property
    def qualified_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name


@dataclass
class CodeChunk:
    """A semantic chunk of code with file/line metadata — the retrieval unit."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    text: str
    language: str = "text"
    symbol_name: str | None = None
    symbol_kind: str | None = None
    imports: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CodeChunk":
        return cls(**d)


@dataclass
class SearchResult:
    chunk: CodeChunk
    score: float
    source: str = "hybrid"  # "vector" | "bm25" | "hybrid" | "rerank"


@dataclass
class Citation:
    """A grounded source reference attached to an answer."""

    ref: str  # e.g. "auth/service.py:42-91" or "PR #421" or "commit abc1234"
    kind: str  # "code" | "pr" | "issue" | "commit" | "memory"

    def __str__(self) -> str:
        return self.ref
