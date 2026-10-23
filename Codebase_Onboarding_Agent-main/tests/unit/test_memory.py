from onboarding_agent.memory.store import MemoryStore, extract_concepts


def make_store() -> MemoryStore:
    return MemoryStore("sqlite:///:memory:")


def test_extract_concepts_removes_stopwords():
    concepts = extract_concepts("How does the authentication token flow work?")
    assert "authentication" in concepts
    assert "token" in concepts
    assert "the" not in concepts and "how" not in concepts


def test_record_and_search_interaction():
    store = make_store()
    store.record_interaction(
        "u1", "How does authentication work?", "It uses tokens [auth/tokens.py:1-10]",
        citations=["auth/tokens.py:1-10"], files=["auth/tokens.py"],
    )
    hits = store.search("u1", "authentication tokens")
    assert len(hits) == 1
    assert "authentication" in hits[0]["question"].lower()
    assert hits[0]["citations"] == ["auth/tokens.py:1-10"]
    assert store.search("u2", "authentication") == []  # isolated per user


def test_familiarity_increases_with_repetition():
    store = make_store()
    for _ in range(3):
        store.record_interaction("u1", "explain the token flow", "answer")
    fam = store.familiarity("u1", ["token"])
    assert 0.4 < fam["token"] <= 1.0
    store.record_interaction("u1", "more about token internals", "answer")
    assert store.familiarity("u1", ["token"])["token"] > fam["token"]


def test_familiarity_summary_splits_known_and_new():
    store = make_store()
    for _ in range(5):
        store.record_interaction("u1", "authentication token details", "a")
    summary = store.familiarity_summary("u1", "authentication and websocket handling")
    assert "authentication" in summary["familiar_concepts"]
    assert "websocket" in summary["new_concepts"]
    assert summary["previously_discussed"]
