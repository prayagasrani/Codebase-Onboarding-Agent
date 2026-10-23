"""Composition root: wires settings into a ready-to-use application service."""

from __future__ import annotations

from pathlib import Path

from onboarding_agent.agent.graph import OnboardingAgent
from onboarding_agent.agent.llm import build_llm
from onboarding_agent.agent.tools import ToolExecutor
from onboarding_agent.config import Settings, get_settings
from onboarding_agent.github.client import GitHubClient
from onboarding_agent.indexing.incremental import IncrementalReport, incremental_update
from onboarding_agent.indexing.indexer import RepoIndex
from onboarding_agent.ingestion.repo import RepoIngestor
from onboarding_agent.logging import configure_logging, get_logger
from onboarding_agent.memory.store import MemoryStore
from onboarding_agent.retrieval.embeddings import build_embedder
from onboarding_agent.retrieval.reranker import LexicalOverlapReranker, LLMReranker, NoopReranker
from onboarding_agent.retrieval.retriever import HybridRetriever

log = get_logger(__name__)


class AppService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        configure_logging(self.settings.log_level)

        self.embedder = build_embedder(
            self.settings.embedding_provider,
            self.settings.embedding_dim,
            self.settings.embedding_model,
        )
        self.index = RepoIndex.load(self.settings.index_dir, self.embedder)
        self.memory = MemoryStore(self.settings.database_url)
        self.llm = build_llm(self.settings)
        self.github = self._build_github()
        self.retriever = self._build_retriever()

    # ------------------------------------------------------------------ wiring
    def _build_github(self) -> GitHubClient | None:
        repo = self.settings.github_repo
        if not repo and self.index.manifest.get("repo_root"):
            repo = RepoIngestor.git_remote_repo(Path(self.index.manifest["repo_root"])) or ""
        if not repo:
            log.info("github_disabled_no_repo")
            return None
        try:
            return GitHubClient(repo, self.settings.github_token)
        except ValueError as exc:
            log.warning("github_disabled", error=str(exc))
            return None

    def _build_retriever(self) -> HybridRetriever:
        if self.settings.reranker == "llm" and self.llm is not None:
            reranker = LLMReranker(self.llm)
        elif self.settings.reranker == "llm":
            reranker = LexicalOverlapReranker()
        else:
            reranker = NoopReranker()
        return HybridRetriever(
            embedder=self.embedder,
            vector_store=self.index.vector_store,
            bm25_store=self.index.bm25_store,
            chunks=self.index.chunks,
            rrf_k=self.settings.rrf_k,
            reranker=reranker,
        )

    def build_agent(self, user_id: str = "default", update_memory: bool = True) -> OnboardingAgent:
        executor = ToolExecutor(
            index=self.index,
            retriever=self.retriever,
            memory=self.memory,
            user_id=user_id,
            github=self.github,
        )
        return OnboardingAgent(
            executor=executor,
            llm=self.llm,
            github_available=self.github is not None,
            update_memory=update_memory,
        )

    # ------------------------------------------------------------------ operations
    def index_repo(self, repo: str) -> dict:
        """Index a local path or clone+index a GitHub URL, then persist."""
        if repo.startswith(("http://", "https://", "git@")):
            dest = Path("./data/repos") / repo.rstrip("/").split("/")[-1].removesuffix(".git")
            root = RepoIngestor().clone(repo, dest)
        else:
            root = Path(repo)
        if not root.exists():
            raise FileNotFoundError(f"Repository path not found: {root}")
        stats = self.index.build(root)
        self.index.save()
        self.retriever = self._build_retriever()
        self.github = self._build_github()
        return stats

    def refresh_index(self) -> IncrementalReport:
        """Incremental re-index of the already-indexed repository."""
        root = self.index.manifest.get("repo_root")
        if not root:
            raise RuntimeError("No repository indexed yet — call index_repo first")
        report = incremental_update(self.index, Path(root))
        if report.changed:
            self.index.save()
            self.retriever = self._build_retriever()
        return report

    def ask(self, question: str, user_id: str = "default") -> dict:
        if not self.index.is_built:
            raise RuntimeError("No repository indexed yet — call index_repo first")
        agent = self.build_agent(user_id=user_id)
        return agent.ask(question, user_id=user_id)
