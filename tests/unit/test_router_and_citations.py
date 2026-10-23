from onboarding_agent.agent.citations import extract_citations, verify_citations
from onboarding_agent.agent.router import QueryRouter


def test_router_code_questions():
    r = QueryRouter()
    assert r.route("How does the LoginService class authenticate users?") == "code"
    assert r.route("Where is the retry logic implemented?") == "code"


def test_router_github_questions():
    r = QueryRouter()
    assert r.route("What open PRs touch the billing module?") in ("github", "mixed")
    assert r.route("Is the CI pipeline passing right now?") in ("github", "mixed")


def test_router_memory_questions():
    r = QueryRouter()
    assert r.route("What did I ask about caching last time?") == "memory"
    assert r.route("Remind me what we discussed about tokens") == "memory"


def test_router_mixed():
    r = QueryRouter()
    route = r.route("Which recent commits changed the authenticate function?")
    assert route == "mixed"


def test_router_defaults_to_code_when_ambiguous():
    assert QueryRouter().route("tell me about payments") == "code"


def test_extract_citations_kinds():
    answer = (
        "Login happens in [auth/service.py:12-25] which calls issue_token "
        "[auth/tokens.py:8-14]. This is being refactored in [PR #421], see also "
        "[commit abc1234] and [issue #7]. Not a citation: [some vague thing!!]."
    )
    cites = extract_citations(answer)
    refs = {c.ref: c.kind for c in cites}
    assert refs["auth/service.py:12-25"] == "code"
    assert refs["PR #421"] == "pr"
    assert refs["commit abc1234"] == "commit"
    assert refs["issue #7"] == "issue"
    assert "some vague thing!!" not in refs


def test_verify_citations_supports_subranges():
    cites = extract_citations("See [auth/service.py:14-16] and [auth/other.py:1-5] and [PR #9]")
    evidence = ["auth/service.py:10-30", "PR #9"]
    v = verify_citations(cites, evidence)
    assert "auth/service.py:14-16" in v["supported"]
    assert "PR #9" in v["supported"]
    assert "auth/other.py:1-5" in v["unsupported"]
    assert 0 < v["support_rate"] < 1


def test_verify_citations_empty_answer_is_clean():
    assert verify_citations([], ["a.py:1-2"])["support_rate"] == 1.0


def test_unicode_dash_citations_extracted_and_verified():
    answer = "RRF lives in fusion [retrieval/fusion.py:8Xi19] and is used by the retriever."
    answer = answer.replace("Xi", "\u2011")  # non-breaking hyphen as emitted by open models
    cites = extract_citations(answer)
    assert [c.ref for c in cites] == ["retrieval/fusion.py:8-19"]
    result = verify_citations(cites, ["retrieval/fusion.py:1-30"])
    assert result["support_rate"] == 1.0


def test_partial_path_citation_supported_at_component_boundary():
    cites = extract_citations("see [retriever.py:40-50] and [ever.py:40-50]")
    result = verify_citations(cites, ["retrieval/retriever.py:36-63"])
    assert result["supported"] == ["retriever.py:40-50"]  # suffix on a "/" boundary
    assert result["unsupported"] == ["ever.py:40-50"]     # substring is not enough
