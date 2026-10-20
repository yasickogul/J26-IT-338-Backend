"""Frozen shared corpus baseline; never imports mutable application models."""
from alembic import op

revision = "shared_0001"
down_revision = None
branch_labels = ("shared",)
depends_on = None

# Frozen PostgreSQL DDL, tested against ORM compilation.
STATEMENTS = [
    "\nCREATE TABLE shared.legal_documents (\n\tid UUID DEFAULT gen_random_uuid() NOT NULL, \n\tsource_id TEXT NOT NULL, \n\tsource_type TEXT NOT NULL, \n\ttitle TEXT NOT NULL, \n\tcitation TEXT, \n\tcourt TEXT, \n\tcase_type TEXT, \n\tdecision_date DATE, \n\tlanguage TEXT, \n\tsource_url TEXT, \n\tfull_text TEXT, \n\tmetadata JSONB DEFAULT '{}'::jsonb NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id), \n\tCONSTRAINT legal_documents_source_nonblank CHECK (btrim(source_id) <> ''), \n\tCONSTRAINT legal_documents_metadata_object CHECK (jsonb_typeof(metadata) = 'object'), \n\tUNIQUE (source_id)\n)\n\n",
    "\nCREATE TABLE shared.legal_chunks (\n\tid UUID DEFAULT gen_random_uuid() NOT NULL, \n\tdocument_id UUID NOT NULL, \n\tchunk_id TEXT NOT NULL, \n\toriginal_chunk_id TEXT NOT NULL, \n\tchunk_index INTEGER NOT NULL, \n\tsection TEXT, \n\ttext TEXT NOT NULL, \n\tembedding_text TEXT NOT NULL, \n\tmetadata JSONB NOT NULL, \n\tparent_context JSONB, \n\tparagraph TEXT, \n\tembedding_model TEXT NOT NULL, \n\tembedding VECTOR(768) NOT NULL, \n\ttsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english'::regconfig, text)) STORED, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, \n\tPRIMARY KEY (id), \n\tCONSTRAINT legal_chunks_document_position UNIQUE (document_id, chunk_index), \n\tCONSTRAINT legal_chunks_nonnegative_index CHECK (chunk_index >= 0), \n\tCONSTRAINT legal_chunks_metadata_object CHECK (jsonb_typeof(metadata) = 'object'), \n\tCONSTRAINT legal_chunks_context_object CHECK (parent_context IS NULL OR jsonb_typeof(parent_context) = 'object'), \n\tCONSTRAINT legal_chunks_chunk_id_nonblank CHECK (btrim(chunk_id) <> ''), \n\tCONSTRAINT legal_chunks_original_chunk_id_nonblank CHECK (btrim(original_chunk_id) <> ''), \n\tCONSTRAINT legal_chunks_text_nonblank CHECK (btrim(text) <> ''), \n\tCONSTRAINT legal_chunks_embedding_text_nonblank CHECK (btrim(embedding_text) <> ''), \n\tCONSTRAINT legal_chunks_embedding_model_nonblank CHECK (btrim(embedding_model) <> ''), \n\tFOREIGN KEY(document_id) REFERENCES shared.legal_documents (id) ON DELETE CASCADE, \n\tUNIQUE (chunk_id)\n)\n\n",
    "CREATE INDEX ix_legal_chunks_document ON shared.legal_chunks (document_id)",
    "CREATE INDEX ix_legal_chunks_model ON shared.legal_chunks (embedding_model)",
    "CREATE INDEX ix_legal_chunks_parent ON shared.legal_chunks ((metadata ->> 'source_id'), (metadata ->> 'section_label'))",
    "CREATE INDEX ix_legal_chunks_tsv ON shared.legal_chunks USING gin (tsv)"
]

def upgrade():
    # Refuse missing extension and existing tables; do not adopt or overwrite.
    op.execute("""DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
            RAISE EXCEPTION 'pgvector must be provisioned before shared_0001';
        END IF;
        IF to_regclass('shared.legal_documents') IS NOT NULL
           OR to_regclass('shared.legal_chunks') IS NOT NULL THEN
            RAISE EXCEPTION 'Existing corpus objects require explicit reconciliation';
        END IF;
    END $$""")
    op.execute("CREATE SCHEMA IF NOT EXISTS shared")
    for statement in STATEMENTS:
        op.execute(statement)

def downgrade():
    # Refuse data loss. No CASCADE; leave shared schema and extension untouched.
    op.execute("LOCK TABLE shared.legal_chunks, shared.legal_documents IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM shared.legal_chunks)
           OR EXISTS (SELECT 1 FROM shared.legal_documents) THEN
            RAISE EXCEPTION 'Refusing to drop nonempty corpus tables';
        END IF;
    END $$""")
    op.execute("DROP TABLE shared.legal_chunks")
    op.execute("DROP TABLE shared.legal_documents")
