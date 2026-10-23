"""Shared fixtures: a small sample repository and an in-memory service."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from onboarding_agent.config import Settings
from onboarding_agent.service import AppService

SAMPLE_FILES = {
    "auth/service.py": '''
        """Authentication service."""
        from auth.tokens import issue_token, verify_token


        class LoginService:
            """Handles user login and session creation."""

            def __init__(self, user_store):
                self.user_store = user_store

            def authenticate(self, username, password):
                """Check credentials and return a session token."""
                user = self.user_store.find(username)
                if user is None or not user.check_password(password):
                    raise PermissionError("invalid credentials")
                return issue_token(user.id)

            def validate_session(self, token):
                return verify_token(token)
    ''',
    "auth/tokens.py": '''
        """Token helpers used by the login flow."""
        import hashlib
        import time

        SECRET = "not-a-real-secret"


        def issue_token(user_id):
            """Create a signed session token for a user."""
            payload = f"{user_id}:{int(time.time())}"
            sig = hashlib.sha256((payload + SECRET).encode()).hexdigest()[:16]
            return f"{payload}:{sig}"


        def verify_token(token):
            """Validate a token issued by issue_token."""
            payload, _, sig = token.rpartition(":")
            expected = hashlib.sha256((payload + SECRET).encode()).hexdigest()[:16]
            return sig == expected
    ''',
    "api/login.py": '''
        """HTTP login endpoint."""
        from auth.service import LoginService


        def login_route(request, service: LoginService):
            """POST /login — authenticates and sets the session cookie."""
            token = service.authenticate(request["username"], request["password"])
            return {"status": 200, "token": token}
    ''',
    "README.md": """
        # Sample App

        A tiny sample application used to test the onboarding agent.
        Authentication lives in `auth/`, HTTP endpoints in `api/`.
    """,
}


@pytest.fixture()
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "sample_repo"
    for rel, content in SAMPLE_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return root


@pytest.fixture()
def service(sample_repo: Path, tmp_path: Path) -> AppService:
    settings = Settings(
        groq_api_key="",
        github_token="",
        github_repo="",
        database_url="sqlite:///:memory:",
        index_dir=tmp_path / "index",
        embedding_provider="hash",
        embedding_dim=256,
        log_level="WARNING",
    )
    svc = AppService(settings)
    svc.index_repo(str(sample_repo))
    return svc
