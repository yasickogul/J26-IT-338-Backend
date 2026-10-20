"""SQLAlchemy models for schema `c1` (Stage 2 intake + extractions)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Text, func
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class C1Base(DeclarativeBase):
    pass


class CaseIntake(C1Base):
    __tablename__ = "case_intakes"
    __table_args__ = {"schema": "c1"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="queued")
    dispute_category: Mapped[str | None] = mapped_column(Text)
    filename: Mapped[str | None] = mapped_column(Text)
    mime: Mapped[str | None] = mapped_column(Text)
    storage_path: Mapped[str | None] = mapped_column(Text)
    intake_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, server_default=sql_text("'{}'::jsonb")
    )
    raw_text: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    extraction: Mapped[Extraction | None] = relationship(
        back_populates="intake", uselist=False, cascade="all, delete-orphan"
    )


class Extraction(C1Base):
    __tablename__ = "extractions"
    __table_args__ = {"schema": "c1"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )
    intake_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("c1.case_intakes.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    structured_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, server_default=sql_text("'{}'::jsonb")
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    intake: Mapped[CaseIntake] = relationship(back_populates="extraction")


# Alias expected by SETUP_AND_RUN_GUIDE / Alembic notes
Base = C1Base
