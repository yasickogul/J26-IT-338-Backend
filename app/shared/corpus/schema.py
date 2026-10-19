"""DDL helpers for the shared legal corpus (Neon / pgvector)."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

CORPUS_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS shared;

CREATE TABLE IF NOT EXISTS shared.legal_documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_type TEXT NOT NULL,
  title TEXT NOT NULL,
  citation TEXT,
  court TEXT,
  case_type TEXT,
  decision_date DATE,
  language TEXT DEFAULT 'en',
  source_url TEXT,
  full_text TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shared.legal_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES shared.legal_documents(id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  section TEXT,
  text TEXT NOT NULL,
  tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  embedding VECTOR(384),
  metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS legal_chunks_tsv_idx
  ON shared.legal_chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS legal_chunks_document_id_idx
  ON shared.legal_chunks (document_id);
CREATE UNIQUE INDEX IF NOT EXISTS legal_documents_citation_uidx
  ON shared.legal_documents (citation)
  WHERE citation IS NOT NULL;
"""

# HNSW needs real vectors; create after Stage 1 embeddings are filled.
EMBEDDING_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS legal_chunks_embedding_idx
  ON shared.legal_chunks USING hnsw (embedding vector_cosine_ops)
"""


async def ensure_corpus_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await _run_ddl(conn)


async def ensure_embedding_dimension(engine: AsyncEngine, dim: int) -> dict:
    """Alter shared.legal_chunks.embedding to VECTOR(dim) when it differs.

    Safe while all (or most) embeddings are NULL; drops the HNSW index first if present.
    """
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod) AS col_type
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    JOIN pg_namespace n ON c.relnamespace = n.oid
                    WHERE n.nspname = 'shared'
                      AND c.relname = 'legal_chunks'
                      AND a.attname = 'embedding'
                      AND a.attnum > 0
                      AND NOT a.attisdropped
                    """
                )
            )
        ).mappings().first()

        if row is None:
            raise RuntimeError("shared.legal_chunks.embedding column not found")

        col_type = row["col_type"]  # e.g. vector(768)
        target = f"vector({dim})"
        if col_type == target:
            return {"changed": False, "from": col_type, "to": target}

        await conn.execute(text("DROP INDEX IF EXISTS shared.legal_chunks_embedding_idx"))
        await conn.execute(
            text(f"ALTER TABLE shared.legal_chunks ALTER COLUMN embedding TYPE vector({dim})")
        )
        return {"changed": True, "from": col_type, "to": target}


async def ensure_embedding_index(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(text(EMBEDDING_INDEX_DDL.strip()))


async def _run_ddl(conn: AsyncConnection) -> None:
    # Run statement-by-statement so a partial failure is easier to diagnose.
    for stmt in CORPUS_DDL.strip().split(";"):
        sql = stmt.strip()
        if sql:
            await conn.execute(text(sql))
