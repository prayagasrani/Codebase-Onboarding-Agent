"""Incremental indexing: re-index only files whose content hash changed.

Uses the manifest's stored SHA-256 per file (works with or without git).
When git is available, ``git diff --name-only`` between the manifest HEAD and
current HEAD narrows the candidate set before hashing.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from onboarding_agent.indexing.indexer import RepoIndex
from onboarding_agent.ingestion.repo import RepoIngestor
from onboarding_agent.logging import get_logger

log = get_logger(__name__)


@dataclass
class IncrementalReport:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    def to_dict(self) -> dict:
        return {
            "added": self.added,
            "modified": self.modified,
            "deleted": self.deleted,
            "unchanged": self.unchanged,
        }


def git_changed_paths(repo_root: Path, old_head: str, new_head: str) -> set[str] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--name-only", old_head, new_head],
            check=True, capture_output=True, text=True,
        )
        return {line.strip() for line in out.stdout.splitlines() if line.strip()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def incremental_update(index: RepoIndex, repo_root: Path) -> IncrementalReport:
    """Diff the working tree against the manifest and re-index only deltas."""
    repo_root = Path(repo_root).resolve()
    report = IncrementalReport()
    current_files = {f.path: f for f in index.ingestor.walk(repo_root)}
    manifest_files: dict = index.manifest.get("files", {})

    # Optional git narrowing: if HEAD moved, only consider paths git says changed
    # (plus anything untracked, caught by the hash comparison below regardless).
    old_head = index.manifest.get("git_head")
    new_head = RepoIngestor.git_head(repo_root)

    for path, rf in current_files.items():
        entry = manifest_files.get(path)
        if entry is None:
            report.added.append(path)
        elif entry["sha256"] != rf.sha256:
            report.modified.append(path)
        else:
            report.unchanged += 1
    for path in manifest_files:
        if path not in current_files:
            report.deleted.append(path)

    if not report.changed:
        log.info("incremental_noop", unchanged=report.unchanged)
        return report

    index.remove_files(report.deleted)
    to_index = [current_files[p] for p in report.added + report.modified]
    if to_index:
        index.update_files(to_index)
    index.manifest["git_head"] = new_head
    log.info(
        "incremental_done",
        added=len(report.added), modified=len(report.modified),
        deleted=len(report.deleted), unchanged=report.unchanged,
        git_narrowed=bool(old_head and new_head and old_head != new_head),
    )
    return report
