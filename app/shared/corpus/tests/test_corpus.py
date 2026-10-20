"""Offline contract checks for the shared MiniLM legal corpus.

C3's separate LaBSE/768-dimensional artifact validation and migration tests
live in app.components.c3_misinformation.tests. Shared corpus tests must not
expect C3-specific source IDs or embeddings in shared.legal_chunks.
"""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from app.core.config import settings
from app.shared.corpus.ingest import (
    citation_exists,
    insert_judgment,
    parse_decision_date,
)
from app.shared.corpus.models import LegalChunk, LegalDocument, SharedBase
from app.shared.corpus.schema import CORPUS_DDL, EMBEDDING_INDEX_DDL


def test_shared_orm_owns_its_tables_without_mixing_component_metadata():
    assert LegalDocument.metadata is SharedBase.metadata
    assert LegalChunk.metadata is SharedBase.metadata
    assert {table.fullname for table in SharedBase.metadata.tables.values()} == {
        "shared.legal_documents",
        "shared.legal_chunks",
    }
    assert LegalDocument.__table__.schema == "shared"
    assert LegalChunk.__table__.schema == "shared"


def test_shared_legal_document_columns_and_chunk_relationship():
    documents = LegalDocument.__table__
    chunks = LegalChunk.__table__
    assert {"id", "source_type", "title", "citation", "full_text", "metadata"}.issubset(
        documents.c.keys()
    )
    assert {"id", "document_id", "chunk_index", "text", "embedding", "metadata"}.issubset(
        chunks.c.keys()
    )
    assert isinstance(documents.c.metadata.type, JSONB)
    assert isinstance(chunks.c.metadata.type, JSONB)
    assert next(iter(chunks.c.document_id.foreign_keys)).target_fullname == (
        "shared.legal_documents.id"
    )
    assert next(iter(chunks.c.document_id.foreign_keys)).ondelete == "CASCADE"


def test_shared_embedding_uses_its_own_configured_dimension():
    embedding = LegalChunk.__table__.c.embedding
    assert isinstance(embedding.type, Vector)
    assert embedding.type.dim == settings.embedding_dim
    assert embedding.nullable  # Documents can be ingested before embedding.
    assert "VECTOR(384)" in CORPUS_DDL
    assert "VECTOR(768)" not in CORPUS_DDL
    assert "USING hnsw" in EMBEDDING_INDEX_DDL


def test_shared_ddl_retains_keyword_index_and_judgment_provenance():
    assert "CREATE TABLE IF NOT EXISTS shared.legal_documents" in CORPUS_DDL
    assert "CREATE TABLE IF NOT EXISTS shared.legal_chunks" in CORPUS_DDL
    assert "GENERATED ALWAYS AS" in CORPUS_DDL
    assert "USING GIN (tsv)" in CORPUS_DDL
    assert "ON DELETE CASCADE" in CORPUS_DDL
    assert "legal_documents_citation_uidx" in CORPUS_DDL


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-10-20", date(2026, 10, 20)),
        ("20/10/2026", date(2026, 10, 20)),
        ("20-10-2026", date(2026, 10, 20)),
        ("2026/10/20", date(2026, 10, 20)),
        ("not a date", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_decision_date(value, expected):
    assert parse_decision_date(value) == expected


@pytest.mark.asyncio
async def test_citation_exists_uses_shared_documents():
    session = MagicMock()
    found = MagicMock()
    found.scalar_one_or_none.return_value = uuid.uuid4()
    session.execute = AsyncMock(return_value=found)
    assert await citation_exists(session, "CA/2026/1") is True
    statement = session.execute.await_args.args[0]
    assert "shared.legal_documents" in str(statement)
    assert "citation" in str(statement)

    found.scalar_one_or_none.return_value = None
    assert await citation_exists(session, "unseen") is False


@pytest.mark.asyncio
async def test_insert_judgment_preserves_metadata_and_chunks_without_embeddings():
    session = MagicMock()
    session.add = MagicMock()
    identity = uuid.uuid4()

    async def flush_document():
        session.add.call_args_list[0].args[0].id = identity

    session.flush = AsyncMock(side_effect=flush_document)
    with patch(
        "app.shared.corpus.ingest.chunk_text",
        return_value=["First passage", "Second passage"],
    ) as chunker:
        document, count = await insert_judgment(
            session,
            title="Synthetic judgment",
            citation="CA/TEST/2026",
            court="Court of Appeal",
            case_type="civil",
            decision_date=date(2026, 10, 20),
            source_url="https://example.test/judgment",
            full_text="Synthetic case facts",
            metadata={"synthetic": True},
        )

    assert count == 2
    assert document.id == identity
    assert document.title == "Synthetic judgment"
    assert document.metadata_ == {"synthetic": True}
    assert document.source_type == "judgment"
    assert document.language == "en"
    assert session.add.call_count == 3
    session.flush.assert_awaited_once()
    chunker.assert_called_once_with(
        "Synthetic case facts", target_tokens=400, overlap_ratio=0.15
    )
    for index, call in enumerate(session.add.call_args_list[1:]):
        chunk = call.args[0]
        assert isinstance(chunk, LegalChunk)
        assert chunk.document_id == identity
        assert chunk.chunk_index == index
        assert chunk.embedding is None
        assert chunk.metadata_ == {}
        assert chunk.text in {"First passage", "Second passage"}
