"""Offline C3-only migration review: never connect, stamp, or run Alembic upgrade."""
import importlib.util
import io
import os
import re
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[1] / "migrations"
TEAM = ROOT / "team"
REVISION = TEAM / "versions/c3_team_0001_claim_lifecycle.py"
CORPUS_REVISION = TEAM / "versions/c3_team_0002_legal_corpus.py"


def load_revision(path):
    spec = importlib.util.spec_from_file_location("reviewed_revision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_body(path, operation="upgrade"):
    output = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        getattr(load_revision(path), operation)()
    return output.getvalue()


def render_review_sql():
    """Render proposed first application, including schema-local tracking DDL."""
    output = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output},
    )
    with context.begin_transaction(), Operations.context(context) as op:
        op.create_table(
            "alembic_version_c3", sa.Column("version_num", sa.String(32), nullable=False),
            sa.PrimaryKeyConstraint("version_num", name="alembic_version_c3_pkc"),
            schema="c3",
        )
        load_revision(REVISION).upgrade()
        op.execute("INSERT INTO c3.alembic_version_c3 (version_num) VALUES ('c3_team_0001') RETURNING c3.alembic_version_c3.version_num")
    return output.getvalue()


class TeamMigrationTests(unittest.TestCase):
    def test_independent_graph_has_no_shared_baseline(self):
        script = ScriptDirectory.from_config(Config(str(TEAM / "alembic.ini")))
        self.assertEqual(script.get_heads(), ["c3_team_0002"])
        self.assertEqual(
            [r.revision for r in script.walk_revisions()],
            ["c3_team_0002", "c3_team_0001"],
        )
        self.assertIsNone(script.get_revision("c3_team_0001").down_revision)
        self.assertEqual(
            script.get_revision("c3_team_0002").down_revision,
            "c3_team_0001",
        )

    def test_lifecycle_ddl_matches_original_frozen_migration(self):
        for operation in ("upgrade", "downgrade"):
            with self.subTest(operation=operation):
                self.assertEqual(
                    render_body(REVISION, operation),
                    render_body(ROOT / "versions/c3_0001_claim_lifecycle.py", operation),
                )

    def test_review_sql_only_creates_c3_objects_and_matches_deliverable(self):
        sql = render_review_sql()
        self.assertEqual(set(re.findall(r"CREATE TABLE ([\w.]+)", sql)), {
            "c3.alembic_version_c3", "c3.claims", "c3.verification_results",
        })
        self.assertNotRegex(sql, r"\b(?:shared|c1|c2|public)\.")
        self.assertNotRegex(sql, r"\b(?:DROP|TRUNCATE|ALTER|UPDATE)\b")
        self.assertNotRegex(sql, r"\bDELETE\s+FROM\b")
        self.assertIn("FOREIGN KEY(claim_id) REFERENCES c3.claims", sql)
        self.assertIn("UNIQUE (claim_id)", sql)
        self.assertIn("WHERE anonymous_session_hash IS NOT NULL", sql)
        self.assertEqual((TEAM / "reviewed_upgrade.sql").read_text(), sql)

    def test_downgrade_refuses_nonempty_tables_and_keeps_schema(self):
        sql = render_body(REVISION, "downgrade")
        self.assertIn("LOCK TABLE c3.verification_results, c3.claims IN ACCESS EXCLUSIVE MODE", sql)
        self.assertIn("Refusing to drop nonempty C3 lifecycle tables", sql)
        self.assertLess(sql.index("RAISE EXCEPTION"), sql.index("DROP TABLE"))
        self.assertNotIn("DROP SCHEMA", sql)
        self.assertNotIn("CASCADE", sql)

    def test_corpus_revision_creates_only_c3_objects_and_preserves_data(self):
        sql = render_body(CORPUS_REVISION)
        self.assertEqual(
            set(re.findall(r"CREATE TABLE ([\w.]+)", sql)),
            {"c3.legal_sources", "c3.legal_chunks"},
        )
        self.assertIn("VECTOR(768) NOT NULL", sql)
        self.assertIn("REFERENCES c3.legal_sources", sql)
        self.assertNotRegex(sql, r"\b(?:shared|c1|c2|public)\.")
        self.assertNotRegex(sql, r"\b(?:DROP|TRUNCATE|ALTER|UPDATE)\b")
        self.assertNotRegex(sql, r"\bDELETE\s+FROM\b")
        downgrade = render_body(CORPUS_REVISION, "downgrade")
        self.assertIn("Refusing to drop nonempty C3 legal corpus tables", downgrade)
        self.assertNotIn("DROP SCHEMA", downgrade)

    def test_corpus_ddl_matches_canonical_migration(self):
        canonical = ROOT / "versions/c3_0002_legal_corpus.py"
        for operation in ("upgrade", "downgrade"):
            with self.subTest(operation=operation):
                self.assertEqual(
                    render_body(CORPUS_REVISION, operation),
                    render_body(canonical, operation),
                )

    def env_namespace(self):
        # Load the environment's helpers through a mocked offline context only.
        with patch("alembic.context.is_offline_mode", return_value=True), \
             patch("alembic.context.configure"), \
             patch("alembic.context.begin_transaction"), \
             patch("alembic.context.run_migrations"):
            return runpy.run_path(str(TEAM / "env.py"))

    def test_online_configuration_refuses_wrong_target_and_unsafe_urls(self):
        approved_url = self.env_namespace()["approved_url"]
        host = "ep-fixture-pooler.example.neon.tech"
        valid = f"postgresql+asyncpg://fixture:secret@{host}/neondb?ssl=verify-full"
        cases = ["", valid.replace(host, "other.example.neon.tech"),
                 valid.replace("/neondb", "/another_db"),
                 valid.replace("verify-full", "require"),
                 valid + "&channel_binding=require", valid.replace("+asyncpg", "")]
        for value in cases:
            with self.subTest(value=value), patch.dict(os.environ, {
                "C3_MIGRATION_URL": value, "C3_MIGRATION_EXPECTED_HOST": host,
            }, clear=True):
                with self.assertRaises(RuntimeError) as caught:
                    approved_url()
                self.assertNotIn("secret", str(caught.exception))
        with patch.dict(os.environ, {
            "C3_MIGRATION_URL": valid, "C3_MIGRATION_EXPECTED_HOST": host,
        }, clear=True):
            self.assertEqual(approved_url().host, host)

    def test_preflight_refuses_missing_schema_wrong_database_or_global_tracking(self):
        migrate = self.env_namespace()["migrate"]
        from unittest.mock import MagicMock
        for scalar_values in (["wrong_db"], ["neondb", False], ["neondb", True, "alembic_version"]):
            connection = MagicMock()
            connection.scalar.side_effect = scalar_values
            with self.subTest(values=scalar_values), self.assertRaises(RuntimeError):
                migrate(connection)


if __name__ == "__main__":
    unittest.main()
