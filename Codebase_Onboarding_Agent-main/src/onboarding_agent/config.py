"""Application settings loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (Groq, OpenAI-compatible API). Leave the key empty for offline/extractive mode.
    llm_max_tokens: int = 4096
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # GitHub
    github_token: str = ""
    github_repo: str = ""  # "owner/name"

    # Storage
    database_url: str = "sqlite:///./data/memory.db"
    index_dir: Path = Path("./data/index")

    # Retrieval
    embedding_provider: str = "hash"  # "hash" | "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 512
    retrieval_top_k: int = 10
    rrf_k: int = 60
    reranker: str = "none"  # "none" | "llm"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.groq_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
