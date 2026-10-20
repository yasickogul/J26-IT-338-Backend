"""Independent C3-only baseline for an existing team database.

Frozen lifecycle DDL copied from c3_0001, without its shared_0001 dependency.
This history owns only C3 tables; it does not assert shared corpus compatibility.
Do not apply both lifecycle histories to the same database.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "c3_team_0001"
down_revision = None
branch_labels = ("c3_team",)
depends_on = None


def upgrade() -> None:
    # Never adopt an object whose ownership or schema is unknown.
    op.execute("""DO $$ BEGIN
        IF to_regclass('c3.claims') IS NOT NULL
           OR to_regclass('c3.verification_results') IS NOT NULL THEN
            RAISE EXCEPTION 'Existing C3 objects require explicit reconciliation';
        END IF;
    END $$""")
    op.execute("CREATE SCHEMA IF NOT EXISTS c3")

    op.create_table(
        "claims",
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("input_type", sa.String(length=10), nullable=False),
        sa.Column("raw_input_text", sa.Text(), nullable=True),
        sa.Column("cleaned_text", sa.Text(), nullable=True),
        sa.Column("detected_language", sa.String(length=10), nullable=True),
        sa.Column("extracted_claim", sa.Text(), nullable=True),
        sa.Column("legal_domain", sa.String(length=30), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "processing_status", sa.String(length=16), nullable=False,
            server_default=sa.text("'queued'"),
        ),
        sa.Column(
            "queued_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_category", sa.String(length=32), nullable=True),
        sa.Column("worker_token", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("anonymous_session_hash", postgresql.CHAR(length=64), nullable=True),
        sa.PrimaryKeyConstraint("claim_id"),
        sa.CheckConstraint(
            "input_type IN ('text', 'image')", name="claims_input_type",
        ),
        sa.CheckConstraint(
            "input_type <> 'text' OR "
            "(raw_input_text IS NOT NULL AND raw_input_text !~ '^[[:space:]]*$')",
            name="claims_text_raw_input_nonblank",
        ),
        sa.CheckConstraint(
            "processing_status IN ('queued', 'processing', 'completed', 'failed')",
            name="claims_processing_status_check",
        ),
        sa.CheckConstraint(
            "failure_category IS NULL OR failure_category IN ("
            "'input', 'input_too_long', 'request_too_large', 'retrieval', "
            "'preparation', 'provider', 'response_validation', 'service', "
            "'result_storage', 'interrupted')",
            name="claims_failure_category_check",
        ),
        sa.CheckConstraint(
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
        sa.CheckConstraint(
            "(processing_status = 'processing' AND "
            "((worker_token IS NULL AND lease_expires_at IS NULL) OR "
            "(worker_token IS NOT NULL AND lease_expires_at IS NOT NULL))) "
            "OR (processing_status <> 'processing' AND "
            "worker_token IS NULL AND lease_expires_at IS NULL)",
            name="claims_worker_lease_shape_check",
        ),
        sa.CheckConstraint(
            "anonymous_session_hash IS NULL OR anonymous_session_hash ~ '^[0-9a-f]{64}$'",
            name="claims_anonymous_session_hash_format",
        ),
        schema="c3",
    )
    op.create_index(
        "claims_anonymous_session_hash_idx",
        "claims",
        ["anonymous_session_hash"],
        unique=False,
        schema="c3",
        postgresql_where=sa.text("anonymous_session_hash IS NOT NULL"),
    )

    op.create_table(
        "verification_results",
        sa.Column("result_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("confidence_score", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("llm_model_name", sa.String(length=100), nullable=True),
        sa.Column("retrieval_top_k", sa.SmallInteger(), nullable=False),
        sa.Column("result_payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("result_id"),
        sa.ForeignKeyConstraint(
            ["claim_id"], ["c3.claims.claim_id"],
            name="verification_results_claim_id_fkey",
        ),
        sa.UniqueConstraint("claim_id", name="verification_results_claim_id_key"),
        sa.CheckConstraint(
            "verdict IN ('True', 'False', 'Misleading', 'insufficient_evidence')",
            name="verification_results_verdict_check",
        ),
        sa.CheckConstraint(
            "confidence_score BETWEEN 0 AND 1",
            name="verification_results_confidence_check",
        ),
        sa.CheckConstraint(
            "retrieval_top_k BETWEEN 1 AND 50",
            name="verification_results_top_k_check",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(result_payload) = 'object'",
            name="verification_results_payload_object_check",
        ),
        schema="c3",
    )


def downgrade() -> None:
    # Never silently discard lifecycle or result data.  Leave the C3 schema
    # itself in place because it may contain objects owned by another revision.
    op.execute("LOCK TABLE c3.verification_results, c3.claims IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM c3.verification_results)
           OR EXISTS (SELECT 1 FROM c3.claims) THEN
            RAISE EXCEPTION 'Refusing to drop nonempty C3 lifecycle tables';
        END IF;
    END $$""")
    op.drop_table("verification_results", schema="c3")
    op.drop_index(
        "claims_anonymous_session_hash_idx", table_name="claims", schema="c3",
    )
    op.drop_table("claims", schema="c3")
