"""BM25 lexical index over code-aware tokens."""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from onboarding_agent.retrieval.tokenize import tokenize


class BM25Store:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunk_ids: list[str], texts: list[str]) -> None:
        for cid, text in zip(chunk_ids, texts, strict=True):
            self._ids.append(cid)
            self._corpus_tokens.append(tokenize(text))
        self._rebuild()

    def remove(self, chunk_ids: set[str]) -> None:
        if not chunk_ids:
            return
        keep = [i for i, cid in enumerate(self._ids) if cid not in chunk_ids]
        self._ids = [self._ids[i] for i in keep]
        self._corpus_tokens = [self._corpus_tokens[i] for i in keep]
        self._rebuild()

    def _rebuild(self) -> None:
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        q_tokens = tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        q_set = set(q_tokens)
        # On tiny corpora BM25 IDF can be zero or negative for real matches,
        # so require token overlap rather than a positive score.
        hits: list[tuple[str, float]] = []
        for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
            if scores[i] > 0 or q_set & set(self._corpus_tokens[i]):
                hits.append((self._ids[i], float(scores[i])))
                if len(hits) >= top_k:
                    break
        return hits

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"ids": self._ids, "tokens": self._corpus_tokens}
        (directory / "bm25.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "BM25Store":
        store = cls()
        path = directory / "bm25.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            store._ids = payload["ids"]
            store._corpus_tokens = payload["tokens"]
            store._rebuild()
        return store
