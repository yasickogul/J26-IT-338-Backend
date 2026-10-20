"""Add the independent C3 LaBSE legal corpus without touching shared data."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c3_0002"
down_revision = "c3_0001"
branch_labels = None
depends_on = None


class Vector768(sa.types.UserDefinedType):
    cache_ok = True

    def get_col_spec(self, **_kwargs):
        return "VECTOR(768)"


def upgrade() -> None:
    op.execute("""DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
            RAISE EXCEPTION 'pgvector must be provisioned before the C3 legal corpus migration';
        END IF;
        IF to_regclass('c3.legal_sources') IS NOT NULL
           OR to_regclass('c3.legal_chunks') IS NOT NULL THEN
            RAISE EXCEPTION 'Existing C3 corpus objects require explicit reconciliation';
        END IF;
    END $$""")
    op.create_table(
        "legal_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("legal_domain", sa.Text(), nullable=False),
        sa.Column("document_type", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("citation", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("local_source_path", sa.Text(), nullable=False),
        sa.Column("source_file_sha256", postgresql.CHAR(length=64), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id", name="legal_sources_source_id_key"),
        sa.CheckConstraint("btrim(source_id) <> ''", name="legal_sources_source_id_nonblank"),
        sa.CheckConstraint("btrim(title) <> ''", name="legal_sources_title_nonblank"),
        sa.CheckConstraint("btrim(legal_domain) <> ''", name="legal_sources_domain_nonblank"),
        sa.CheckConstraint("btrim(document_type) <> ''", name="legal_sources_type_nonblank"),
        sa.CheckConstraint("btrim(language) <> ''", name="legal_sources_language_nonblank"),
        sa.CheckConstraint(
            "btrim(local_source_path) <> '' AND local_source_path !~ '^/' "
            "AND local_source_path !~ '(^|/)\\.\\.(/|$)'",
            name="legal_sources_relative_path",
        ),
        sa.CheckConstraint("source_file_sha256 ~ '^[0-9a-f]{64}$'", name="legal_sources_sha256_format"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="legal_sources_metadata_object"),
        schema="c3",
    )
    op.create_table(
        "legal_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("original_chunk_id", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("parent_context", postgresql.JSONB(), nullable=True),
        sa.Column("paragraph", sa.Text(), nullable=True),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("embedding", Vector768(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["source_id"], ["c3.legal_sources.source_id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("chunk_id", name="legal_chunks_chunk_id_key"),
        sa.UniqueConstraint("source_id", "chunk_index", name="legal_chunks_source_position_key"),
        sa.CheckConstraint("btrim(chunk_id) <> ''", name="legal_chunks_chunk_id_nonblank"),
        sa.CheckConstraint("btrim(original_chunk_id) <> ''", name="legal_chunks_original_chunk_id_nonblank"),
        sa.CheckConstraint("chunk_index >= 0", name="legal_chunks_nonnegative_index"),
        sa.CheckConstraint("btrim(text) <> ''", name="legal_chunks_text_nonblank"),
        sa.CheckConstraint("btrim(embedding_text) <> ''", name="legal_chunks_embedding_text_nonblank"),
        sa.CheckConstraint("btrim(embedding_model) <> ''", name="legal_chunks_embedding_model_nonblank"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="legal_chunks_metadata_object"),
        sa.CheckConstraint(
            "parent_context IS NULL OR jsonb_typeof(parent_context) = 'object'",
            name="legal_chunks_parent_context_object",
        ),
        schema="c3",
    )
    op.create_index("ix_c3_legal_chunks_source", "legal_chunks", ["source_id"], schema="c3")
    op.create_index("ix_c3_legal_chunks_embedding_model", "legal_chunks", ["embedding_model"], schema="c3")
    op.execute(
        "CREATE INDEX ix_c3_legal_chunks_parent ON c3.legal_chunks "
        "((metadata ->> 'source_id'), (metadata ->> 'section_label'))"
    )


def downgrade() -> None:
    op.execute("LOCK TABLE c3.legal_chunks, c3.legal_sources IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM c3.legal_chunks)
           OR EXISTS (SELECT 1 FROM c3.legal_sources) THEN
            RAISE EXCEPTION 'Refusing to drop nonempty C3 legal corpus tables';
        END IF;
    END $$""")
    op.drop_table("legal_chunks", schema="c3")
    op.drop_table("legal_sources", schema="c3")
