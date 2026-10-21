-- Additive C4-only embedding lifecycle metadata.
-- Execute through the approved migration process after local tests pass.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
DECLARE
    embedding_type TEXT;
BEGIN
    SELECT format_type(a.atttypid, a.atttypmod)
    INTO embedding_type
    FROM pg_attribute a
    JOIN pg_class c ON c.oid = a.attrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'c_argumentation'
      AND c.relname = 'judgment_chunks'
      AND a.attname = 'embedding'
      AND a.attisdropped = false;

    IF embedding_type <> 'vector(384)' THEN
        RAISE EXCEPTION
            'Expected c_argumentation.judgment_chunks.embedding to be vector(384), found %',
            embedding_type;
    END IF;
END
$$;

ALTER TABLE c_argumentation.judgment_chunks
    ADD COLUMN IF NOT EXISTS embedding_model TEXT;

ALTER TABLE c_argumentation.judgment_chunks
    ADD COLUMN IF NOT EXISTS embedding_status TEXT NOT NULL DEFAULT 'pending';

ALTER TABLE c_argumentation.judgment_chunks
    ADD COLUMN IF NOT EXISTS embedding_source_hash TEXT;

ALTER TABLE c_argumentation.judgment_chunks
    ADD COLUMN IF NOT EXISTS embedding_generated_at TIMESTAMPTZ;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'c4_embedding_status_check'
          AND conrelid = 'c_argumentation.judgment_chunks'::regclass
    ) THEN
        ALTER TABLE c_argumentation.judgment_chunks
            ADD CONSTRAINT c4_embedding_status_check
            CHECK (embedding_status IN ('pending', 'processing', 'ready', 'failed'));
    END IF;
END
$$;

COMMIT;
