"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from onboarding_agent.api.schemas import (
    HealthResponse,
    IndexRequest,
    IndexResponse,
    OnboardRequest,
    OnboardResponse,
    QueryRequest,
    QueryResponse,
    RefreshResponse,
)
from onboarding_agent.logging import get_logger
from onboarding_agent.onboarding.generator import OnboardingGenerator
from onboarding_agent.service import AppService

log = get_logger(__name__)


def create_app(service: AppService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or AppService()
        log.info("api_started", index_built=app.state.service.index.is_built)
        yield

    app = FastAPI(title="Codebase Onboarding Agent", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        svc: AppService = app.state.service
        return HealthResponse(
            status="ok",
            index_built=svc.index.is_built,
            llm_enabled=svc.llm is not None,
            github_enabled=svc.github is not None,
            stats=svc.index.stats() if svc.index.is_built else {},
        )

    @app.post("/index", response_model=IndexResponse)
    def index_repo(req: IndexRequest) -> IndexResponse:
        svc: AppService = app.state.service
        try:
            stats = svc.index_repo(req.repo)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return IndexResponse(stats=stats)

    @app.post("/index/refresh", response_model=RefreshResponse)
    def refresh_index() -> RefreshResponse:
        svc: AppService = app.state.service
        try:
            report = svc.refresh_index()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RefreshResponse(report=report.to_dict())

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        svc: AppService = app.state.service
        try:
            result = svc.ask(req.question, user_id=req.user_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            verification=result["verification"],
            route=result["route"],
            latency_s=result["latency_s"],
            evidence_count=len(result["evidence"]),
        )

    @app.post("/onboard", response_model=OnboardResponse)
    def onboard(req: OnboardRequest) -> OnboardResponse:
        svc: AppService = app.state.service
        if not svc.index.is_built:
            raise HTTPException(status_code=409, detail="No repository indexed yet")
        result = OnboardingGenerator(svc).generate()
        return OnboardResponse(guide=result["guide"], mode=result["mode"], facts=result["facts"])

    return app


app = create_app()


def run() -> None:
    import uvicorn

    from onboarding_agent.config import get_settings

    settings = get_settings()
    uvicorn.run("onboarding_agent.api.main:app", host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    run()
