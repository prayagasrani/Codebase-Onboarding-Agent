from pathlib import Path

from onboarding_agent.indexing.incremental import incremental_update


def test_incremental_noop(service, sample_repo):
    report = incremental_update(service.index, sample_repo)
    assert not report.changed
    assert report.unchanged > 0


def test_incremental_detects_modification(service, sample_repo: Path):
    before_chunks = set(service.index.chunks.keys())
    target = sample_repo / "auth" / "tokens.py"
    target.write_text(
        target.read_text(encoding="utf-8") + "\n\ndef revoke_token(token):\n    return None\n",
        encoding="utf-8",
    )
    report = incremental_update(service.index, sample_repo)
    assert report.modified == ["auth/tokens.py"]
    assert not report.added and not report.deleted
    after_chunks = set(service.index.chunks.keys())
    assert after_chunks != before_chunks
    # New symbol must be searchable and present in the symbol table.
    names = {s.name for s in service.index.symbols["auth/tokens.py"]}
    assert "revoke_token" in names


def test_incremental_detects_add_and_delete(service, sample_repo: Path):
    new_file = sample_repo / "api" / "logout.py"
    new_file.write_text("def logout_route(request):\n    return {'status': 200}\n",
                        encoding="utf-8")
    (sample_repo / "README.md").unlink()
    report = incremental_update(service.index, sample_repo)
    assert "api/logout.py" in report.added
    assert "README.md" in report.deleted
    assert "api/logout.py" in service.index.manifest["files"]
    assert "README.md" not in service.index.manifest["files"]
    # Deleted file's chunks must be gone from retrieval.
    for chunk in service.index.chunks.values():
        assert chunk.file_path != "README.md"
