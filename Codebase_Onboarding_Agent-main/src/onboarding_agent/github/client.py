"""Live GitHub context via the REST API (httpx).

Answers the "what is currently happening in development" side of questions,
as opposed to "what the code currently contains" (the index). All methods
return plain dicts trimmed to the fields the agent needs, each carrying a
``ref`` suitable for citations (e.g. "PR #421", "commit abc1234").
"""

from __future__ import annotations

from typing import Any

import httpx

from onboarding_agent.logging import get_logger

log = get_logger(__name__)

API = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


class GitHubClient:
    def __init__(self, repo: str, token: str = "", client: httpx.Client | None = None):
        """repo: "owner/name". token optional (rate-limited without it)."""
        if "/" not in repo:
            raise ValueError(f"repo must be 'owner/name', got {repo!r}")
        self.repo = repo
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = client or httpx.Client(base_url=API, headers=headers, timeout=20.0)

    def _get(self, endpoint: str, **params: Any) -> Any:
        resp = self._client.get(endpoint, params={k: v for k, v in params.items() if v is not None})
        if resp.status_code >= 400:
            raise GitHubError(f"GitHub API {resp.status_code} for {endpoint}: {resp.text[:300]}")
        return resp.json()

    # ------------------------------------------------------------------ commits
    def commits(self, limit: int = 10, path: str | None = None, sha: str | None = None) -> list[dict]:
        data = self._get(f"/repos/{self.repo}/commits", per_page=limit, path=path, sha=sha)
        return [
            {
                "ref": f"commit {c['sha'][:7]}",
                "sha": c["sha"],
                "message": c["commit"]["message"].split("\n")[0][:200],
                "author": (c["commit"]["author"] or {}).get("name"),
                "date": (c["commit"]["author"] or {}).get("date"),
                "url": c.get("html_url"),
            }
            for c in data
        ]

    def commit(self, sha: str) -> dict:
        c = self._get(f"/repos/{self.repo}/commits/{sha}")
        return {
            "ref": f"commit {c['sha'][:7]}",
            "sha": c["sha"],
            "message": c["commit"]["message"][:1000],
            "author": (c["commit"]["author"] or {}).get("name"),
            "date": (c["commit"]["author"] or {}).get("date"),
            "files": [
                {"filename": f["filename"], "status": f["status"],
                 "additions": f["additions"], "deletions": f["deletions"]}
                for f in c.get("files", [])[:50]
            ],
        }

    def file_history(self, path: str, limit: int = 10) -> list[dict]:
        return self.commits(limit=limit, path=path)

    # ------------------------------------------------------------------ pull requests
    def pull_requests(self, state: str = "open", limit: int = 10) -> list[dict]:
        data = self._get(f"/repos/{self.repo}/pulls", state=state, per_page=limit,
                         sort="updated", direction="desc")
        return [self._pr_summary(pr) for pr in data]

    def pull_request(self, number: int) -> dict:
        pr = self._get(f"/repos/{self.repo}/pulls/{number}")
        files = self._get(f"/repos/{self.repo}/pulls/{number}/files", per_page=50)
        out = self._pr_summary(pr)
        out["body"] = (pr.get("body") or "")[:2000]
        out["files"] = [f["filename"] for f in files]
        return out

    @staticmethod
    def _pr_summary(pr: dict) -> dict:
        return {
            "ref": f"PR #{pr['number']}",
            "number": pr["number"],
            "title": pr["title"],
            "state": pr["state"],
            "draft": pr.get("draft", False),
            "author": (pr.get("user") or {}).get("login"),
            "updated_at": pr.get("updated_at"),
            "head": (pr.get("head") or {}).get("ref"),
            "url": pr.get("html_url"),
        }

    # ------------------------------------------------------------------ issues
    def issues(self, state: str = "open", limit: int = 10, labels: str | None = None) -> list[dict]:
        data = self._get(f"/repos/{self.repo}/issues", state=state, per_page=limit, labels=labels)
        return [
            {
                "ref": f"issue #{i['number']}",
                "number": i["number"],
                "title": i["title"],
                "state": i["state"],
                "author": (i.get("user") or {}).get("login"),
                "labels": [lb["name"] for lb in i.get("labels", [])],
                "updated_at": i.get("updated_at"),
                "url": i.get("html_url"),
            }
            for i in data
            if "pull_request" not in i  # /issues also returns PRs
        ]

    # ------------------------------------------------------------------ branches / CI
    def branches(self, limit: int = 20) -> list[dict]:
        data = self._get(f"/repos/{self.repo}/branches", per_page=limit)
        return [{"ref": f"branch {b['name']}", "name": b["name"], "sha": b["commit"]["sha"][:7]}
                for b in data]

    def ci_status(self, ref: str = "HEAD") -> dict:
        if ref == "HEAD":
            ref = self.default_branch()
        runs = self._get(f"/repos/{self.repo}/commits/{ref}/check-runs", per_page=20)
        checks = [
            {"name": r["name"], "status": r["status"], "conclusion": r.get("conclusion")}
            for r in runs.get("check_runs", [])
        ]
        overall = "success"
        if any(c["status"] != "completed" for c in checks):
            overall = "pending"
        elif any(c["conclusion"] not in ("success", "neutral", "skipped") for c in checks):
            overall = "failure"
        elif not checks:
            overall = "unknown"
        return {"ref": f"CI @ {ref}", "sha_or_branch": ref, "overall": overall, "checks": checks}

    def default_branch(self) -> str:
        return self._get(f"/repos/{self.repo}")["default_branch"]

    # ------------------------------------------------------------------ search
    def search(self, query: str, kind: str = "issues", limit: int = 10) -> list[dict]:
        """kind: 'issues' (issues+PRs), 'commits', or 'code'."""
        q = f"{query} repo:{self.repo}"
        if kind == "commits":
            data = self._get("/search/commits", q=q, per_page=limit)
            return [
                {"ref": f"commit {it['sha'][:7]}",
                 "message": it["commit"]["message"].split("\n")[0][:200]}
                for it in data.get("items", [])
            ]
        if kind == "code":
            data = self._get("/search/code", q=q, per_page=limit)
            return [{"ref": it["path"], "path": it["path"], "url": it.get("html_url")}
                    for it in data.get("items", [])]
        data = self._get("/search/issues", q=q, per_page=limit)
        out = []
        for it in data.get("items", []):
            is_pr = "pull_request" in it
            out.append({
                "ref": f"{'PR' if is_pr else 'issue'} #{it['number']}",
                "number": it["number"],
                "title": it["title"],
                "state": it["state"],
            })
        return out
