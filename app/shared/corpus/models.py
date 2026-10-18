"""Shared corpus: UUID entities and stable textual citation IDs."""
from sqlalchemy import (Column, Table, Text, Integer, Date, DateTime, ForeignKey,
                        CheckConstraint, UniqueConstraint, Index, Computed, text)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import UserDefinedType


class Vector768(UserDefinedType):
    """DDL type; vector I/O uses explicit SQL casts, not an implicit ORM codec."""
    cache_ok = True

    def get_col_spec(self, **kw):
        return "VECTOR(768)"


class Base(DeclarativeBase):
    pass


documents = Table(
    "legal_documents", Base.metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("source_id", Text, nullable=False, unique=True),
    Column("source_type", Text, nullable=False), Column("title", Text, nullable=False),
    Column("citation", Text), Column("court", Text), Column("case_type", Text),
    Column("decision_date", Date), Column("language", Text),
    Column("source_url", Text), Column("full_text", Text),
    Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    CheckConstraint("btrim(source_id) <> ''", name="legal_documents_source_nonblank"),
    CheckConstraint("jsonb_typeof(metadata) = 'object'", name="legal_documents_metadata_object"),
    schema="shared",
)

chunks = Table(
    "legal_chunks", Base.metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("document_id", UUID(as_uuid=True), ForeignKey("shared.legal_documents.id", ondelete="CASCADE"), nullable=False),
    Column("chunk_id", Text, nullable=False, unique=True),
    Column("original_chunk_id", Text, nullable=False),
    Column("chunk_index", Integer, nullable=False), Column("section", Text),
    Column("text", Text, nullable=False), Column("embedding_text", Text, nullable=False),
    Column("metadata", JSONB, nullable=False),
    Column("parent_context", JSONB), Column("paragraph", Text),
    Column("embedding_model", Text, nullable=False),
    Column("embedding", Vector768(), nullable=False),
    Column("tsv", TSVECTOR, Computed("to_tsvector('english'::regconfig, text)", persisted=True)),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("document_id", "chunk_index", name="legal_chunks_document_position"),
    CheckConstraint("chunk_index >= 0", name="legal_chunks_nonnegative_index"),
    CheckConstraint("jsonb_typeof(metadata) = 'object'", name="legal_chunks_metadata_object"),
    CheckConstraint("parent_context IS NULL OR jsonb_typeof(parent_context) = 'object'", name="legal_chunks_context_object"),
    *(CheckConstraint(f"btrim({col}) <> ''", name=f"legal_chunks_{col}_nonblank")
      for col in ("chunk_id", "original_chunk_id", "text", "embedding_text", "embedding_model")),
    schema="shared",
)
Index("ix_legal_chunks_document", chunks.c.document_id)
Index("ix_legal_chunks_model", chunks.c.embedding_model)
Index("ix_legal_chunks_tsv", chunks.c.tsv, postgresql_using="gin")
Index("ix_legal_chunks_parent", chunks.c.metadata["source_id"].astext,
      chunks.c.metadata["section_label"].astext)


class LegalDocument(Base):
    __table__ = documents


class LegalChunk(Base):
    __table__ = chunks
