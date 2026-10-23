"""End-to-end pipeline tests on the sample repo (offline / extractive mode)."""

from pathlib import Path

from onboarding_agent.indexing.indexer import RepoIndex
from onboarding_agent.onboarding.generator import OnboardingGenerator


def test_index_build_stats(service):
    stats = service.index.stats()
    assert stats["files"] == 4
    assert stats["chunks"] > 0
    assert stats["graph_nodes"] > 0


def test_hybrid_retrieval_finds_auth_code(service):
    for mode in ("vector", "bm25", "hybrid", "hybrid_rrf"):
        results = service.retriever.search("how does user authentication work", top_k=5, mode=mode)
        files = {r.chunk.file_path for r in results}
        assert files & {"auth/service.py", "auth/tokens.py"}, f"mode={mode} missed auth files"


def test_graph_traces_dependencies(service):
    result = service.index.graph.trace_dependencies("auth/service.py")
    assert result["found"]
    assert any("auth/tokens.py" in d for d in result["dependencies"])


def test_agent_answers_code_question_with_citations(service):
    result = service.ask("How does the LoginService authenticate users?", user_id="dev1")
    assert result["route"] == "code"
    assert result["answer"]
    assert result["citations"], "extractive answer must cite sources"
    assert any("auth/" in c["ref"] for c in result["citations"])
    # In extractive mode every citation comes from evidence => fully supported.
    assert result["verification"]["support_rate"] == 1.0
    assert result["verification"]["unsupported"] == []


def test_agent_insufficient_evidence_is_explicit(service):
    result = service.ask("How does the kubernetes zebra franchise module work?", user_id="dev1")
    answer = result["answer"].lower()
    found_evidence = bool(result["citations"])
    assert found_evidence or "could not find enough evidence" in answer


def test_memory_persists_across_questions(service):
    service.ask("Explain the token verification flow", user_id="dev2")
    result = service.ask("What did I ask about tokens before?", user_id="dev2")
    assert result["route"] == "memory"
    assert "token" in result["answer"].lower()
    ctx = service.memory.familiarity_summary("dev2", "token verification")
    assert ctx["previously_discussed"]


def test_index_persistence_roundtrip(service, tmp_path: Path):
    service.index.save()
    loaded = RepoIndex.load(service.settings.index_dir, service.embedder)
    assert loaded.is_built
    assert len(loaded.chunks) == len(service.index.chunks)
    assert loaded.graph.g.number_of_nodes() == service.index.graph.g.number_of_nodes()
    assert set(loaded.symbols.keys()) == set(service.index.symbols.keys())


def test_onboarding_guide_offline(service):
    result = OnboardingGenerator(service).generate()
    assert result["mode"] == "template"
    guide = result["guide"]
    for heading in ("Project Overview", "Architecture", "Entry Points",
                    "Major Components", "Testing", "Deployment",
                    "Current Development Activity"):
        assert heading in guide
    assert "auth" in guide  # layout section mentions the auth dir
