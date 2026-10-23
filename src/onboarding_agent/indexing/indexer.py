"""Repository index: builds and persists all retrieval structures.

Persisted artifacts (under ``index_dir``):
  - manifest.json     file -> {sha256, chunk_ids}, plus repo metadata
  - chunks.json       chunk_id -> chunk payload
  - vectors.npy / vector_ids.json (vector store)
  - bm25.json         (lexical store)
  - code_graph.json   (dependency/call graph)

Incremental updates diff current file hashes against the manifest and only
re-parse / re-embed changed files (see ``incremental.py``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from onboarding_agent.graphs.code_graph import CodeGraph
from onboarding_agent.ingestion.repo import RepoFile, RepoIngestor
from onboarding_agent.logging import get_logger
from onboarding_agent.models import CodeChunk, Symbol
from onboarding_agent.parsing.ast_parser import AstParser
from onboarding_agent.parsing.chunker import Chunker
from onboarding_agent.retrieval.bm25_store import BM25Store
from onboarding_agent.retrieval.embeddings import Embedder
from onboarding_agent.retrieval.vector_store import VectorStore

log = get_logger(__name__)


class RepoIndex:
    def __init__(self, index_dir: Path, embedder: Embedder):
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.ingestor = RepoIngestor()
        self.parser = AstParser()
        self.chunker = Chunker()

        self.chunks: dict[str, CodeChunk] = {}
        self.symbols: dict[str, list[Symbol]] = {}  # file -> symbols
        self.vector_store = VectorStore(embedder.dim)
        self.bm25_store = BM25Store()
        self.graph = CodeGraph()
        self.manifest: dict = {"repo_root": "", "git_head": None, "indexed_at": None, "files": {}}

    # ------------------------------------------------------------------ build
    def build(self, repo_root: Path) -> dict:
        """Full index build from scratch."""
        repo_root = Path(repo_root).resolve()
        t0 = time.time()
        files = self.ingestor.walk(repo_root)
        self.chunks.clear()
        self.symbols.clear()
        self.vector_store = VectorStore(self.embedder.dim)
        self.bm25_store = BM25Store()
        self.graph = CodeGraph()
        self.manifest = {
            "repo_root": str(repo_root),
            "git_head": RepoIngestor.git_head(repo_root),
            "indexed_at": time.time(),
            "files": {},
        }
        self._index_files(files, {f.path for f in files})
        self._rebuild_call_edges()
        stats = self.stats()
        stats["elapsed_s"] = round(time.time() - t0, 2)
        log.info("index_built", **stats)
        return stats

    def _index_files(self, files: list[RepoFile], all_paths: set[str]) -> None:
        new_ids: list[str] = []
        new_texts: list[str] = []
        for rf in files:
            text = RepoIngestor.read_text(rf)
            symbols, imports = self.parser.parse(text, rf.path, rf.language)
            chunks = self.chunker.chunk_file(rf.path, text, rf.language, symbols, imports)
            self.symbols[rf.path] = symbols
            self.graph.add_file(rf.path, symbols, imports, all_paths)
            chunk_ids = []
            for chunk in chunks:
                self.chunks[chunk.chunk_id] = chunk
                chunk_ids.append(chunk.chunk_id)
                new_ids.append(chunk.chunk_id)
                new_texts.append(self._embed_text(chunk))
            self.manifest["files"][rf.path] = {"sha256": rf.sha256, "chunk_ids": chunk_ids}
        if new_ids:
            # Batch to bound memory for large repos.
            for i in range(0, len(new_ids), 256):
                batch_ids = new_ids[i:i + 256]
                batch_texts = new_texts[i:i + 256]
                self.vector_store.add(batch_ids, self.embedder.embed(batch_texts))
            self.bm25_store.add(new_ids, new_texts)

    def remove_files(self, paths: list[str]) -> None:
        stale: set[str] = set()
        for path in paths:
            entry = self.manifest["files"].pop(path, None)
            if entry:
                stale |= set(entry["chunk_ids"])
            self.symbols.pop(path, None)
            self.graph.remove_file(path)
        for cid in stale:
            self.chunks.pop(cid, None)
        self.vector_store.remove(stale)
        self.bm25_store.remove(stale)

    def update_files(self, files: list[RepoFile]) -> None:
        """Re-index the given files in place (used by incremental updates)."""
        self.remove_files([f.path for f in files])
        all_paths = set(self.manifest["files"].keys()) | {f.path for f in files}
        self._index_files(files, all_paths)
        self._rebuild_call_edges()
        self.manifest["indexed_at"] = time.time()

    def _rebuild_call_edges(self) -> None:
        symbol_index: dict[str, list[str]] = {}
        for path, syms in self.symbols.items():
            for s in syms:
                symbol_index.setdefault(s.name, []).append(f"sym:{path}::{s.qualified_name}")
        self.graph.add_calls(list(self.chunks.values()), symbol_index)

    @staticmethod
    def _embed_text(chunk: CodeChunk) -> str:
        """Text used for embedding/BM25: path + symbol header improves recall."""
        header = f"{chunk.file_path} {chunk.symbol_name or ''} {chunk.symbol_kind or ''}"
        return f"{header}\n{chunk.text}"

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.index_dir / "chunks.json").write_text(
            json.dumps({cid: c.to_dict() for cid, c in self.chunks.items()}), encoding="utf-8"
        )
        symbol_payload = {
            path: [s.__dict__ for s in syms] for path, syms in self.symbols.items()
        }
        (self.index_dir / "symbols.json").write_text(json.dumps(symbol_payload), encoding="utf-8")
        self.vector_store.save(self.index_dir)
        self.bm25_store.save(self.index_dir)
        self.graph.save(self.index_dir)
        log.info("index_saved", dir=str(self.index_dir), chunks=len(self.chunks))

    @classmethod
    def load(cls, index_dir: Path, embedder: Embedder) -> "RepoIndex":
        index = cls(index_dir, embedder)
        manifest_path = index.index_dir / "manifest.json"
        if not manifest_path.exists():
            return index
        index.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks_path = index.index_dir / "chunks.json"
        if chunks_path.exists():
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            index.chunks = {cid: CodeChunk.from_dict(d) for cid, d in payload.items()}
        symbols_path = index.index_dir / "symbols.json"
        if symbols_path.exists():
            payload = json.loads(symbols_path.read_text(encoding="utf-8"))
            index.symbols = {
                path: [Symbol(**s) for s in syms] for path, syms in payload.items()
            }
        index.vector_store = VectorStore.load(index.index_dir, embedder.dim)
        index.bm25_store = BM25Store.load(index.index_dir)
        index.graph = CodeGraph.load(index.index_dir)
        return index

    @property
    def is_built(self) -> bool:
        return bool(self.chunks)

    def stats(self) -> dict:
        return {
            "files": len(self.manifest["files"]),
            "chunks": len(self.chunks),
            "vectors": len(self.vector_store),
            "graph_nodes": self.graph.g.number_of_nodes(),
            "graph_edges": self.graph.g.number_of_edges(),
        }
