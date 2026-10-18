"""Offline structural checks for C3-owned PostgreSQL models."""

import unittest

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Numeric, SmallInteger, String, Text, inspect
from sqlalchemy.dialects.postgresql import CHAR, JSONB, UUID, dialect
from sqlalchemy.schema import CreateIndex, CreateTable

from app.components.c3_misinformation.models import Base, Claim, VerificationResult


class C3ModelTests(unittest.TestCase):
    def setUp(self):
        self.claims = Claim.__table__
        self.results = VerificationResult.__table__

    def test_only_c3_claim_and_result_tables_are_registered(self):
        self.assertEqual(set(Base.metadata.tables), {"c3.claims", "c3.verification_results"})
        self.assertEqual((self.claims.name, self.claims.schema), ("claims", "c3"))
        self.assertEqual((self.results.name, self.results.schema), ("verification_results", "c3"))
        self.assertIs(Claim.metadata, VerificationResult.metadata)

    def test_claim_columns_match_compnet_schema(self):
        self.assertEqual(list(self.claims.columns.keys()), [
            "claim_id", "input_type", "raw_input_text", "cleaned_text",
            "detected_language", "extracted_claim", "legal_domain", "created_at",
            "processing_status", "queued_at", "started_at", "completed_at",
            "failed_at", "failure_category", "worker_token", "lease_expires_at",
            "anonymous_session_hash",
        ])
        for name in ("claim_id", "worker_token"):
            self.assertIsInstance(self.claims.c[name].type, UUID)
            self.assertTrue(self.claims.c[name].type.as_uuid)
        for name, length in (("input_type", 10), ("detected_language", 10),
                             ("legal_domain", 30), ("processing_status", 16),
                             ("failure_category", 32)):
            self.assertIsInstance(self.claims.c[name].type, String)
            self.assertEqual(self.claims.c[name].type.length, length)
        for name in ("raw_input_text", "cleaned_text", "extracted_claim"):
            self.assertIsInstance(self.claims.c[name].type, Text)
        self.assertIsInstance(self.claims.c.anonymous_session_hash.type, CHAR)
        self.assertEqual(self.claims.c.anonymous_session_hash.type.length, 64)
        for name in ("created_at", "queued_at", "started_at", "completed_at",
                     "failed_at", "lease_expires_at"):
            self.assertIsInstance(self.claims.c[name].type, DateTime)
            self.assertTrue(self.claims.c[name].type.timezone)

    def test_claim_nullability_and_server_defaults(self):
        for name in ("claim_id", "input_type", "created_at", "processing_status", "queued_at"):
            self.assertFalse(self.claims.c[name].nullable, name)
        for name in ("raw_input_text", "cleaned_text", "detected_language", "extracted_claim",
                     "legal_domain", "started_at", "completed_at", "failed_at",
                     "failure_category", "worker_token", "lease_expires_at", "anonymous_session_hash"):
            self.assertTrue(self.claims.c[name].nullable, name)
        self.assertEqual(str(self.claims.c.processing_status.server_default.arg), "'queued'")
        for name in ("created_at", "queued_at"):
            self.assertEqual(str(self.claims.c[name].server_default.arg), "CURRENT_TIMESTAMP")
        self.assertIsNone(self.claims.c.claim_id.default)
        self.assertIsNone(self.claims.c.worker_token.server_default)
        self.assertIsNone(self.claims.c.lease_expires_at.server_default)

    def test_claim_checks_preserve_states_failures_lease_and_hash(self):
        checks = {item.name: str(item.sqltext) for item in self.claims.constraints
                  if isinstance(item, CheckConstraint)}
        self.assertEqual(set(checks), {
            "claims_input_type", "claims_text_raw_input_nonblank",
            "claims_processing_status_check", "claims_failure_category_check",
            "claims_lifecycle_shape_check", "claims_worker_lease_shape_check",
            "claims_anonymous_session_hash_format",
        })
        self.assertIn("('queued', 'processing', 'completed', 'failed')",
                      checks["claims_processing_status_check"])
        self.assertIn("'interrupted'", checks["claims_failure_category_check"])
        self.assertIn("started_at IS NOT NULL", checks["claims_lifecycle_shape_check"])
        self.assertIn("worker_token IS NOT NULL AND lease_expires_at IS NOT NULL",
                      checks["claims_worker_lease_shape_check"])
        self.assertIn("^[0-9a-f]{64}$", checks["claims_anonymous_session_hash_format"])
        self.assertIn("raw_input_text !~ '^[[:space:]]*$'",
                      checks["claims_text_raw_input_nonblank"])

    def test_anonymous_owner_index_is_partial_and_no_raw_token_is_stored(self):
        self.assertEqual(len(self.claims.indexes), 1)
        index = next(iter(self.claims.indexes))
        self.assertEqual(index.name, "claims_anonymous_session_hash_idx")
        self.assertEqual([column.name for column in index.columns], ["anonymous_session_hash"])
        ddl = str(CreateIndex(index).compile(dialect=dialect()))
        self.assertIn("ON c3.claims (anonymous_session_hash)", ddl)
        self.assertIn("WHERE anonymous_session_hash IS NOT NULL", ddl)
        self.assertNotIn("session_token", self.claims.columns)
        self.assertNotIn("user_id", self.claims.columns)

    def test_result_columns_types_nullability_and_defaults(self):
        self.assertEqual(list(self.results.columns.keys()), [
            "result_id", "claim_id", "verdict", "confidence_score", "explanation",
            "llm_model_name", "retrieval_top_k", "result_payload", "created_at",
        ])
        for name in ("result_id", "claim_id"):
            self.assertIsInstance(self.results.c[name].type, UUID)
            self.assertFalse(self.results.c[name].nullable)
        self.assertIsInstance(self.results.c.verdict.type, String)
        self.assertEqual(self.results.c.verdict.type.length, 32)
        self.assertIsInstance(self.results.c.confidence_score.type, Numeric)
        self.assertEqual((self.results.c.confidence_score.type.precision,
                          self.results.c.confidence_score.type.scale), (4, 3))
        self.assertIsInstance(self.results.c.explanation.type, Text)
        self.assertEqual(self.results.c.llm_model_name.type.length, 100)
        self.assertIsInstance(self.results.c.retrieval_top_k.type, SmallInteger)
        self.assertIsInstance(self.results.c.result_payload.type, JSONB)
        self.assertFalse(self.results.c.result_payload.nullable)
        self.assertTrue(self.results.c.confidence_score.nullable)
        self.assertIsNone(self.results.c.confidence_score.server_default)
        self.assertTrue(self.results.c.explanation.nullable)
        self.assertTrue(self.results.c.llm_model_name.nullable)
        self.assertEqual(str(self.results.c.created_at.server_default.arg), "CURRENT_TIMESTAMP")

    def test_one_result_per_claim_has_schema_local_foreign_key_and_unique_constraint(self):
        self.assertEqual(len(self.results.c.claim_id.foreign_keys), 1)
        foreign_key = next(iter(self.results.c.claim_id.foreign_keys))
        self.assertIsInstance(foreign_key, ForeignKey)
        self.assertEqual(foreign_key.target_fullname, "c3.claims.claim_id")
        self.assertIs(foreign_key.column, self.claims.c.claim_id)
        self.assertTrue(self.results.c.claim_id.unique)
        self.assertFalse(inspect(Claim).relationships["verification_result"].uselist)
        self.assertEqual(inspect(VerificationResult).relationships["claim"].direction.name, "MANYTOONE")

    def test_result_constraints_and_compiled_postgresql_ddl(self):
        checks = {item.name: str(item.sqltext) for item in self.results.constraints
                  if isinstance(item, CheckConstraint)}
        self.assertEqual(set(checks), {
            "verification_results_verdict_check", "verification_results_confidence_check",
            "verification_results_top_k_check", "verification_results_payload_object_check",
        })
        self.assertIn("'insufficient_evidence'", checks["verification_results_verdict_check"])
        self.assertIn("confidence_score BETWEEN 0 AND 1",
                      checks["verification_results_confidence_check"])
        self.assertIn("jsonb_typeof(result_payload) = 'object'",
                      checks["verification_results_payload_object_check"])
        claim_ddl = str(CreateTable(self.claims).compile(dialect=dialect()))
        result_ddl = str(CreateTable(self.results).compile(dialect=dialect()))
        self.assertIn("CREATE TABLE c3.claims", claim_ddl)
        self.assertIn("CREATE TABLE c3.verification_results", result_ddl)
        self.assertIn("result_payload JSONB NOT NULL", result_ddl)
        self.assertIn("FOREIGN KEY(claim_id) REFERENCES c3.claims (claim_id)", result_ddl)
        self.assertIn("UNIQUE (claim_id)", result_ddl)
        self.assertIn("TIMESTAMP WITH TIME ZONE", claim_ddl)


if __name__ == "__main__":
    unittest.main()
