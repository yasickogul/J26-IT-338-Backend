"""SQLAlchemy models for shared.legal_documents and shared.legal_chunks (§5.4)."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.embeddings import DEFAULT_EMBEDDING_DIM
from app.core.config import settings


class SharedBase(DeclarativeBase):
    """Own metadata/base so components never mix schemas."""


_EMBEDDING_DIM = settings.embedding_dim or DEFAULT_EMBEDDING_DIM


class LegalDocument(SharedBase):
    __tablename__ = "legal_documents"
    __table_args__ = {"schema": "shared"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )
    source_type: Mapped[str] = mapped_column(Text, nullable=False)  # act | judgment | regulation
    title: Mapped[str] = mapped_column(Text, nullable=False)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    court: Mapped[Optional[str]] = mapped_column(Text)
    case_type: Mapped[Optional[str]] = mapped_column(Text)
    decision_date: Mapped[Optional[date]] = mapped_column(Date)
    language: Mapped[str] = mapped_column(Text, server_default=sql_text("'en'"))
    source_url: Mapped[Optional[str]] = mapped_column(Text)
    full_text: Mapped[Optional[str]] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=sql_text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["LegalChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class LegalChunk(SharedBase):
    __tablename__ = "legal_chunks"
    __table_args__ = {"schema": "shared"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("shared.legal_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[Optional[str]] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # tsv is a GENERATED ALWAYS column created in DDL — not mapped for writes
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(_EMBEDDING_DIM), nullable=True
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=sql_text("'{}'::jsonb")
    )

    document: Mapped[LegalDocument] = relationship(back_populates="chunks")
