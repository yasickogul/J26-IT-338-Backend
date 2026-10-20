"""C3-owned claim lifecycle and verification-result table definitions."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import CHAR, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import UserDefinedType


class Vector768(UserDefinedType):
    """C3-owned pgvector DDL type; SQL paths use explicit vector casts."""

    cache_ok = True

    def get_col_spec(self, **_kwargs: Any) -> str:
        return "VECTOR(768)"


class Base(DeclarativeBase):
    """Metadata owned by the C3 PostgreSQL schema."""


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (
        CheckConstraint("input_type IN ('text', 'image')", name="claims_input_type"),
        CheckConstraint(
            "input_type <> 'text' OR "
            "(raw_input_text IS NOT NULL AND raw_input_text !~ '^[[:space:]]*$')",
            name="claims_text_raw_input_nonblank",
        ),
        CheckConstraint(
            "processing_status IN ('queued', 'processing', 'completed', 'failed')",
            name="claims_processing_status_check",
        ),
        CheckConstraint(
            "failure_category IS NULL OR failure_category IN ("
            "'input', 'input_too_long', 'request_too_large', 'retrieval', "
            "'preparation', 'provider', 'response_validation', 'service', "
            "'result_storage', 'interrupted')",
            name="claims_failure_category_check",
        ),
        CheckConstraint(
            "(processing_status = 'queued' AND started_at IS NULL "
            "AND completed_at IS NULL AND failed_at IS NULL AND failure_category IS NULL) "
            "OR (processing_status = 'processing' AND started_at IS NOT NULL "
            "AND completed_at IS NULL AND failed_at IS NULL AND failure_category IS NULL) "
            "OR (processing_status = 'completed' AND started_at IS NOT NULL "
            "AND completed_at IS NOT NULL AND failed_at IS NULL AND failure_category IS NULL) "
            "OR (processing_status = 'failed' AND completed_at IS NULL "
            "AND failed_at IS NOT NULL AND failure_category IS NOT NULL)",
            name="claims_lifecycle_shape_check",
        ),
        CheckConstraint(
            "(processing_status = 'processing' AND "
            "((worker_token IS NULL AND lease_expires_at IS NULL) OR "
            "(worker_token IS NOT NULL AND lease_expires_at IS NOT NULL))) "
            "OR (processing_status <> 'processing' AND "
            "worker_token IS NULL AND lease_expires_at IS NULL)",
            name="claims_worker_lease_shape_check",
        ),
        CheckConstraint(
            "anonymous_session_hash IS NULL OR anonymous_session_hash ~ '^[0-9a-f]{64}$'",
            name="claims_anonymous_session_hash_format",
        ),
        Index(
            "claims_anonymous_session_hash_idx", "anonymous_session_hash",
            postgresql_where=sql_text("anonymous_session_hash IS NOT NULL"),
        ),
        {"schema": "c3"},
    )

    claim_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    input_type: Mapped[str] = mapped_column(String(10), nullable=False)
    raw_input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    cleaned_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    detected_language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    extracted_claim: Mapped[str | None] = mapped_column(Text, nullable=True)
    legal_domain: Mapped[str | None] = mapped_column(String(30), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sql_text("CURRENT_TIMESTAMP"),
    )
    processing_status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=sql_text("'queued'"),
    )
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sql_text("CURRENT_TIMESTAMP"),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    worker_token: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    anonymous_session_hash: Mapped[str | None] = mapped_column(CHAR(64), nullable=True)

    verification_result: Mapped[VerificationResult | None] = relationship(
        back_populates="claim", uselist=False,
    )


class VerificationResult(Base):
    __tablename__ = "verification_results"
    __table_args__ = (
        CheckConstraint(
            "verdict IN ('True', 'False', 'Misleading', 'insufficient_evidence')",
            name="verification_results_verdict_check",
        ),
        CheckConstraint(
            "confidence_score BETWEEN 0 AND 1",
            name="verification_results_confidence_check",
        ),
        CheckConstraint(
            "retrieval_top_k BETWEEN 1 AND 50",
            name="verification_results_top_k_check",
        ),
        CheckConstraint(
            "jsonb_typeof(result_payload) = 'object'",
            name="verification_results_payload_object_check",
        ),
        {"schema": "c3"},
    )

    result_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    claim_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("c3.claims.claim_id"), nullable=False, unique=True,
    )
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    retrieval_top_k: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    result_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sql_text("CURRENT_TIMESTAMP"),
    )

    claim: Mapped[Claim] = relationship(back_populates="verification_result")


class LegalSource(Base):
    """Stable C3 citation identity for one locally reviewed legal source."""

    __tablename__ = "legal_sources"
    __table_args__ = (
        CheckConstraint("btrim(source_id) <> ''", name="legal_sources_source_id_nonblank"),
        CheckConstraint("btrim(title) <> ''", name="legal_sources_title_nonblank"),
        CheckConstraint("btrim(legal_domain) <> ''", name="legal_sources_domain_nonblank"),
        CheckConstraint("btrim(document_type) <> ''", name="legal_sources_type_nonblank"),
        CheckConstraint("btrim(language) <> ''", name="legal_sources_language_nonblank"),
        CheckConstraint(
            "btrim(local_source_path) <> '' AND local_source_path !~ '^/' "
            "AND local_source_path !~ '(^|/)\\.\\.(/|$)'",
            name="legal_sources_relative_path",
        ),
        CheckConstraint(
            "source_file_sha256 ~ '^[0-9a-f]{64}$'",
            name="legal_sources_sha256_format",
        ),
        CheckConstraint(
            "jsonb_typeof(metadata) = 'object'",
            name="legal_sources_metadata_object",
        ),
        {"schema": "c3"},
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"),
    )
    source_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    legal_domain: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)
    citation: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    local_source_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_file_sha256: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=sql_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sql_text("CURRENT_TIMESTAMP"),
    )
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sql_text("CURRENT_TIMESTAMP"),
    )

    chunks: Mapped[list[LegalChunk]] = relationship(back_populates="source")


class LegalChunk(Base):
    """One immutable C3 legal passage with LaBSE/768 provenance."""

    __tablename__ = "legal_chunks"
    __table_args__ = (
        CheckConstraint("btrim(chunk_id) <> ''", name="legal_chunks_chunk_id_nonblank"),
        CheckConstraint(
            "btrim(original_chunk_id) <> ''",
            name="legal_chunks_original_chunk_id_nonblank",
        ),
        CheckConstraint("chunk_index >= 0", name="legal_chunks_nonnegative_index"),
        CheckConstraint("btrim(text) <> ''", name="legal_chunks_text_nonblank"),
        CheckConstraint(
            "btrim(embedding_text) <> ''",
            name="legal_chunks_embedding_text_nonblank",
        ),
        CheckConstraint(
            "btrim(embedding_model) <> ''",
            name="legal_chunks_embedding_model_nonblank",
        ),
        CheckConstraint(
            "jsonb_typeof(metadata) = 'object'",
            name="legal_chunks_metadata_object",
        ),
        CheckConstraint(
            "parent_context IS NULL OR jsonb_typeof(parent_context) = 'object'",
            name="legal_chunks_parent_context_object",
        ),
        Index("ix_c3_legal_chunks_source", "source_id"),
        Index("ix_c3_legal_chunks_embedding_model", "embedding_model"),
        UniqueConstraint(
            "source_id", "chunk_index", name="legal_chunks_source_position_key",
        ),
        {"schema": "c3"},
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"),
    )
    source_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("c3.legal_sources.source_id", ondelete="RESTRICT"),
        nullable=False,
    )
    chunk_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    original_chunk_id: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False)
    parent_context: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    paragraph: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector768(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sql_text("CURRENT_TIMESTAMP"),
    )
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sql_text("CURRENT_TIMESTAMP"),
    )

    source: Mapped[LegalSource] = relationship(back_populates="chunks")


Index(
    "ix_c3_legal_chunks_parent",
    LegalChunk.__table__.c.metadata["source_id"].astext,
    LegalChunk.__table__.c.metadata["section_label"].astext,
)
