"""SQLAlchemy models for the C4 legal argumentation corpus."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

# C4 owns a 384-dimensional vector column.  Keep this local rather than
# inheriting the shared application's currently configured 768 dimensions.
_EMBEDDING_DIM = 384


class C4Base(DeclarativeBase):
    """Separate metadata for C4; it is never used to drop other schemas."""


class CourtCase(C4Base):
    __tablename__ = "court_cases"
    __table_args__ = (
        UniqueConstraint("source_document_key", name="uq_c4_court_cases_source_document_key"),
        {"schema": "c_argumentation"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    case_number: Mapped[str] = mapped_column(Text, nullable=False)
    source_document_key: Mapped[str] = mapped_column(Text, nullable=False)
    date_decided: Mapped[Optional[date]] = mapped_column(Date)
    parties: Mapped[Optional[str]] = mapped_column(Text)
    judge: Mapped[Optional[str]] = mapped_column(Text)
    keywords: Mapped[Optional[str]] = mapped_column(Text)
    legislation: Mapped[Optional[str]] = mapped_column(Text)
    case_type: Mapped[Optional[str]] = mapped_column(Text)
    pdf_url: Mapped[Optional[str]] = mapped_column(Text)
    source_csv: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    judgment: Mapped[Optional["CaseJudgment"]] = relationship(back_populates="court_case", uselist=False, cascade="all, delete-orphan")


class CaseJudgment(C4Base):
    __tablename__ = "case_judgments"
    __table_args__ = {"schema": "c_argumentation"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    case_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("c_argumentation.court_cases.id", ondelete="CASCADE"), nullable=False)
    full_text: Mapped[Optional[str]] = mapped_column(Text)
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    extraction_status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    extraction_method: Mapped[Optional[str]] = mapped_column(Text)
    extracted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    court_case: Mapped[CourtCase] = relationship(back_populates="judgment")
    chunks: Mapped[list["JudgmentChunk"]] = relationship(back_populates="judgment", cascade="all, delete-orphan")


class JudgmentChunk(C4Base):
    __tablename__ = "judgment_chunks"
    __table_args__ = (
        UniqueConstraint("judgment_id", "chunk_index", name="uq_c4_judgment_chunks_order"),
        {"schema": "c_argumentation"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    judgment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("c_argumentation.case_judgments.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_page: Mapped[Optional[int]] = mapped_column(Integer)
    end_page: Mapped[Optional[int]] = mapped_column(Integer)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(_EMBEDDING_DIM))
    embedding_model: Mapped[Optional[str]] = mapped_column(Text)
    embedding_status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    embedding_source_hash: Mapped[Optional[str]] = mapped_column(Text)
    embedding_generated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    judgment: Mapped[CaseJudgment] = relationship(back_populates="chunks")
