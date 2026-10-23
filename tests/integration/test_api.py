"""API tests via FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient

from onboarding_agent.api.main import create_app


@pytest.fixture()
def client(service):
    app = create_app(service)
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index_built"] is True
    assert body["llm_enabled"] is False


def test_query_endpoint(client):
    resp = client.post("/query", json={"question": "How does authentication work?",
                                       "user_id": "api-user"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["route"] == "code"
    assert isinstance(body["citations"], list)
    assert body["evidence_count"] > 0


def test_index_endpoint_missing_path(client):
    resp = client.post("/index", json={"repo": "Z:/does/not/exist"})
    assert resp.status_code == 404


def test_refresh_endpoint(client):
    resp = client.post("/index/refresh")
    assert resp.status_code == 200
    assert resp.json()["report"]["unchanged"] > 0


def test_onboard_endpoint(client):
    resp = client.post("/onboard", json={"user_id": "api-user"})
    assert resp.status_code == 200
    assert "Project Overview" in resp.json()["guide"]
