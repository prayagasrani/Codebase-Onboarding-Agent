import httpx
import pytest

from onboarding_agent.github.client import GitHubClient, GitHubError


def make_client(handler) -> GitHubClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(base_url="https://api.github.com", transport=transport)
    return GitHubClient("owner/repo", token="t", client=http)


def test_repo_validation():
    with pytest.raises(ValueError):
        GitHubClient("not-a-repo")


def test_commits_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/owner/repo/commits"
        return httpx.Response(200, json=[{
            "sha": "abcdef1234567",
            "commit": {"message": "fix: login bug\n\ndetails",
                       "author": {"name": "Ada", "date": "2026-08-01T00:00:00Z"}},
            "html_url": "https://github.com/owner/repo/commit/abcdef1",
        }])

    commits = make_client(handler).commits(limit=1)
    assert commits[0]["ref"] == "commit abcdef1"
    assert commits[0]["message"] == "fix: login bug"
    assert commits[0]["author"] == "Ada"


def test_pull_request_with_files():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json=[{"filename": "auth/service.py"}])
        return httpx.Response(200, json={
            "number": 421, "title": "Refactor auth", "state": "open", "draft": False,
            "user": {"login": "ada"}, "updated_at": "2026-08-10T00:00:00Z",
            "head": {"ref": "refactor-auth"}, "html_url": "u", "body": "why & what",
        })

    pr = make_client(handler).pull_request(421)
    assert pr["ref"] == "PR #421"
    assert pr["files"] == ["auth/service.py"]
    assert pr["body"] == "why & what"


def test_issues_excludes_prs():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[
            {"number": 1, "title": "bug", "state": "open", "user": {"login": "x"},
             "labels": [{"name": "bug"}], "updated_at": "t", "html_url": "u"},
            {"number": 2, "title": "a pr", "state": "open", "user": {"login": "y"},
             "labels": [], "updated_at": "t", "html_url": "u", "pull_request": {}},
        ])

    issues = make_client(handler).issues()
    assert len(issues) == 1
    assert issues[0]["ref"] == "issue #1"


def test_ci_status_aggregation():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"check_runs": [
            {"name": "tests", "status": "completed", "conclusion": "success"},
            {"name": "lint", "status": "completed", "conclusion": "failure"},
        ]})

    status = make_client(handler).ci_status("main")
    assert status["overall"] == "failure"
    assert len(status["checks"]) == 2


def test_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(GitHubError):
        make_client(handler).commits()
