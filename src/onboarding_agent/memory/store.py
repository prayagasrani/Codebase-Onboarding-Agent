"""Persistent user memory.

Stores past Q&A exchanges, the files/concepts they covered, and a per-user
familiarity score used to adapt answer depth and avoid repeating explanations.
Backed by SQLAlchemy — SQLite by default, PostgreSQL in production.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, or_, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from onboarding_agent.logging import get_logger
from onboarding_agent.memory.models import Base, ConceptFamiliarity, Interaction, utcnow
from onboarding_agent.retrieval.tokenize import tokenize

log = get_logger(__name__)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "how", "what", "where", "why", "does", "do",
    "in", "of", "to", "for", "and", "or", "this", "that", "it", "i", "we", "you",
    "can", "could", "should", "would", "work", "works", "code", "file", "files",
}


def extract_concepts(text: str, max_concepts: int = 8) -> list[str]:
    """Salient concept keys from a question (deduped, stopwords removed)."""
    seen: list[str] = []
    for tok in tokenize(text):
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        if tok not in seen:
            seen.append(tok)
    return seen[:max_concepts]


class MemoryStore:
    def __init__(self, database_url: str):
        engine_kwargs: dict = {"future": True}
        if database_url.startswith("sqlite"):
            # The API serves requests from worker threads; share one connection safely.
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            db_path = database_url.removeprefix("sqlite:///")
            if ":memory:" in database_url:
                engine_kwargs["poolclass"] = StaticPool
            else:
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, **engine_kwargs)
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def _session(self) -> Session:
        return self._session_factory()

    # ------------------------------------------------------------------ write
    def record_interaction(
        self,
        user_id: str,
        question: str,
        answer: str,
        citations: list[str] | None = None,
        files: list[str] | None = None,
        repo: str = "",
    ) -> list[str]:
        concepts = extract_concepts(question)
        with self._session() as s, s.begin():
            s.add(Interaction(
                user_id=user_id, repo=repo, question=question, answer=answer,
                citations="\n".join(citations or []),
                concepts=",".join(concepts),
                files=",".join(files or []),
            ))
            for concept in concepts:
                row = s.execute(
                    select(ConceptFamiliarity).where(
                        ConceptFamiliarity.user_id == user_id,
                        ConceptFamiliarity.repo == repo,
                        ConceptFamiliarity.concept == concept,
                    )
                ).scalar_one_or_none()
                if row is None:
                    s.add(ConceptFamiliarity(
                        user_id=user_id, repo=repo, concept=concept,
                        familiarity=0.2, times_discussed=1,
                    ))
                else:
                    # Asymptotic approach to 1.0 with each discussion.
                    row.familiarity = min(1.0, row.familiarity + (1.0 - row.familiarity) * 0.25)
                    row.times_discussed += 1
                    row.last_discussed = utcnow()
        return concepts

    # ------------------------------------------------------------------ read
    def search(self, user_id: str, query: str, limit: int = 5, repo: str = "") -> list[dict]:
        """Keyword search over past interactions (question/concepts/files)."""
        terms = [t for t in extract_concepts(query, max_concepts=6)]
        with self._session() as s:
            stmt = select(Interaction).where(Interaction.user_id == user_id)
            if repo:
                stmt = stmt.where(Interaction.repo == repo)
            if terms:
                stmt = stmt.where(or_(*[
                    Interaction.question.ilike(f"%{t}%") for t in terms
                ] + [
                    Interaction.concepts.ilike(f"%{t}%") for t in terms
                ]))
            stmt = stmt.order_by(Interaction.created_at.desc()).limit(limit)
            rows = s.execute(stmt).scalars().all()
        return [
            {
                "question": r.question,
                "answer_snippet": r.answer[:500],
                "citations": r.citations.split("\n") if r.citations else [],
                "concepts": r.concepts.split(",") if r.concepts else [],
                "asked_at": r.created_at.isoformat(),
            }
            for r in rows
        ]

    def familiarity(self, user_id: str, concepts: list[str], repo: str = "") -> dict[str, float]:
        if not concepts:
            return {}
        with self._session() as s:
            rows = s.execute(
                select(ConceptFamiliarity).where(
                    ConceptFamiliarity.user_id == user_id,
                    ConceptFamiliarity.repo == repo,
                    ConceptFamiliarity.concept.in_(concepts),
                )
            ).scalars().all()
        return {r.concept: r.familiarity for r in rows}

    def familiarity_summary(self, user_id: str, question: str, repo: str = "") -> dict:
        """Context block the agent uses to adapt depth and skip repeats."""
        concepts = extract_concepts(question)
        known = self.familiarity(user_id, concepts, repo)
        prior = self.search(user_id, question, limit=3, repo=repo)
        avg = sum(known.values()) / len(known) if known else 0.0
        return {
            "concepts": concepts,
            "familiar_concepts": sorted([c for c, f in known.items() if f >= 0.5]),
            "new_concepts": sorted([c for c in concepts if known.get(c, 0.0) < 0.5]),
            "average_familiarity": round(avg, 2),
            "previously_discussed": prior,
        }

    def interaction_count(self, user_id: str, repo: str = "") -> int:
        with self._session() as s:
            stmt = select(Interaction).where(Interaction.user_id == user_id)
            if repo:
                stmt = stmt.where(Interaction.repo == repo)
            return len(s.execute(stmt).scalars().all())
