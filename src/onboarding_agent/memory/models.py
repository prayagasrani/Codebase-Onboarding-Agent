"""SQLAlchemy models for persistent user memory."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Interaction(Base):
    """One question/answer exchange, with the files and concepts it touched."""

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    repo: Mapped[str] = mapped_column(String(256), index=True, default="")
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    citations: Mapped[str] = mapped_column(Text, default="")   # newline-separated refs
    concepts: Mapped[str] = mapped_column(Text, default="")    # comma-separated concept keys
    files: Mapped[str] = mapped_column(Text, default="")       # comma-separated file paths
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ConceptFamiliarity(Base):
    """How familiar a user is with a concept (0=new .. 1=expert), updated as they ask."""

    __tablename__ = "concept_familiarity"
    __table_args__ = (UniqueConstraint("user_id", "repo", "concept", name="uq_user_repo_concept"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    repo: Mapped[str] = mapped_column(String(256), default="")
    concept: Mapped[str] = mapped_column(String(256), index=True)
    familiarity: Mapped[float] = mapped_column(Float, default=0.0)
    times_discussed: Mapped[int] = mapped_column(Integer, default=0)
    last_discussed: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
