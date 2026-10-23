"""Repository ingestion: local paths or GitHub clones, file walking, hashing."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from onboarding_agent.logging import get_logger

log = get_logger(__name__)

LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".md": "markdown",
    ".rst": "markdown",
    ".txt": "text",
    ".toml": "config",
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
    ".cfg": "config",
    ".ini": "config",
    ".sh": "shell",
    ".sql": "sql",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode",
    "data", "eval_results", ".tox", "target", ".mypy_cache", "site-packages",
}

MAX_FILE_BYTES = 1_000_000  # skip files larger than 1MB (generated / vendored)


@dataclass
class RepoFile:
    path: str  # repo-relative, posix-style
    abs_path: Path
    language: str
    sha256: str
    size: int


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class RepoIngestor:
    """Loads a repository from a local path or clones it from a URL."""

    def __init__(self, exclude_dirs: set[str] | None = None):
        self.exclude_dirs = exclude_dirs or DEFAULT_EXCLUDE_DIRS

    def clone(self, url: str, dest: Path, depth: int = 1) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and (dest / ".git").exists():
            log.info("repo_exists_pulling", dest=str(dest))
            subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=True, capture_output=True)
            return dest
        log.info("cloning_repo", url=url, dest=str(dest))
        subprocess.run(["git", "clone", "--depth", str(depth), url, str(dest)], check=True, capture_output=True)
        return dest

    def walk(self, root: Path) -> list[RepoFile]:
        """Walk the repo and return supported source files with content hashes."""
        root = root.resolve()
        files: list[RepoFile] = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel_parts = p.relative_to(root).parts
            if any(part in self.exclude_dirs for part in rel_parts[:-1]):
                continue
            lang = LANGUAGE_BY_EXTENSION.get(p.suffix.lower())
            if lang is None:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES or size == 0:
                continue
            files.append(
                RepoFile(
                    path=p.relative_to(root).as_posix(),
                    abs_path=p,
                    language=lang,
                    sha256=file_sha256(p),
                    size=size,
                )
            )
        log.info("repo_walked", root=str(root), files=len(files))
        return files

    @staticmethod
    def read_text(repo_file: RepoFile) -> str:
        return repo_file.abs_path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def git_head(root: Path) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            )
            return out.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    @staticmethod
    def git_remote_repo(root: Path) -> str | None:
        """Return 'owner/name' inferred from the origin remote, if any."""
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "remote", "get-url", "origin"],
                check=True, capture_output=True, text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        url = out.stdout.strip()
        for prefix in ("git@github.com:", "https://github.com/", "http://github.com/"):
            if url.startswith(prefix):
                tail = url[len(prefix):]
                return tail.removesuffix(".git")
        return None
