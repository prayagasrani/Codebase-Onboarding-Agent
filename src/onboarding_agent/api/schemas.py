"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    repo: str = Field(description="Local path or GitHub URL to index")


class IndexResponse(BaseModel):
    status: str = "ok"
    stats: dict


class RefreshResponse(BaseModel):
    status: str = "ok"
    report: dict


class QueryRequest(BaseModel):
    question: str
    user_id: str = "default"


class CitationOut(BaseModel):
    ref: str
    kind: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    verification: dict
    route: str
    latency_s: float
    evidence_count: int


class OnboardRequest(BaseModel):
    user_id: str = "default"


class OnboardResponse(BaseModel):
    guide: str
    mode: str
    facts: dict


class HealthResponse(BaseModel):
    status: str
    index_built: bool
    llm_enabled: bool
    github_enabled: bool
    stats: dict
