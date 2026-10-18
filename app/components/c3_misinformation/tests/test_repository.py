"""Mocked async repository checks; no PostgreSQL transaction guarantees implied."""

from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock, create_autospec
from uuid import UUID

from sqlalchemy.dialects.postgresql import dialect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.components.c3_misinformation import repository as repo
from app.components.c3_misinformation.anonymous_claim_session import session_token_hash


CLAIM_ID = UUID("03030303-0303-4303-8303-030303030303")
OWNER = session_token_hash("a" * 43)
OTHER = session_token_hash("b" * 43)


class RepositoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = create_autospec(AsyncSession, instance=True)

    def return_row(self, row):
        result = MagicMock()
        result.mappings.return_value.one_or_none.return_value = row
        self.session.execute.return_value = result

    def compiled(self):
        return self.session.execute.call_args.args[0].compile(dialect=dialect())

    def assert_caller_owned_transaction(self):
        self.session.commit.assert_not_called()
        self.session.rollback.assert_not_called()
        self.session.close.assert_not_called()
        self.session.begin.assert_not_called()
        self.session.flush.assert_not_called()

    async def test_creation_preserves_text_hash_and_database_timestamp_defaults(self):
        claim_id = await repo.create_text_claim(
            "  Original claim\n", cleaned_text="Original claim",
            anonymous_session_hash=OWNER, detected_language="si", session=self.session,
        )
        self.assertIsInstance(claim_id, UUID)
        self.assertEqual(claim_id.version, 4)
        self.session.execute.assert_awaited_once()
        compiled = self.compiled()
        self.assertIn("INSERT INTO c3.claims", str(compiled))
        self.assertEqual(compiled.params, {
            "claim_id": claim_id, "input_type": "text", "raw_input_text": "  Original claim\n",
            "cleaned_text": "Original claim", "anonymous_session_hash": OWNER,
            "detected_language": "si", "processing_status": "queued",
        })
        self.assertNotIn("created_at", compiled.params)
        self.assertNotIn("queued_at", compiled.params)
        self.assert_caller_owned_transaction()

    async def test_optional_cleaned_text_and_languages(self):
        for language in (None, "en", "si", "ta", "mixed"):
            with self.subTest(language=language):
                await repo.create_text_claim("text", anonymous_session_hash=OWNER,
                                             detected_language=language, session=self.session)
                self.assertIsNone(self.compiled().params["cleaned_text"])
                self.assertEqual(self.compiled().params["detected_language"], language)

    async def test_invalid_text_and_language_do_not_execute(self):
        for kwargs in ({"raw_input_text": " "}, {"raw_input_text": None},
                       {"cleaned_text": "\t"}, {"cleaned_text": 42},
                       {"detected_language": "fr"}):
            with self.subTest(kwargs=kwargs):
                args = {"raw_input_text": "text", "anonymous_session_hash": OWNER,
                        "session": self.session, **kwargs}
                with self.assertRaises(ValueError):
                    await repo.create_text_claim(**args)
        self.session.execute.assert_not_called()

    async def test_missing_or_invalid_ownership_is_rejected_before_io(self):
        for owner in (None, "", "a" * 43, "A" * 64, "g" * 64, "a" * 63, "a" * 65, 42):
            with self.subTest(owner=owner):
                with self.assertRaisesRegex(ValueError, "anonymous_session_hash must be a SHA-256 hex digest"):
                    await repo.create_text_claim("text", anonymous_session_hash=owner, session=self.session)
                for read in (repo.get_owned_claim_state, repo.get_owned_claim_result):
                    with self.assertRaises(ValueError):
                        await read(CLAIM_ID, owner, session=self.session)
        with self.assertRaises(TypeError):
            await repo.create_text_claim("text", session=self.session)
        self.session.execute.assert_not_called()
        self.assert_caller_owned_transaction()

    async def test_invalid_claim_id_is_rejected_before_io(self):
        for read in (repo.get_owned_claim_state, repo.get_owned_claim_result):
            with self.assertRaisesRegex(ValueError, "claim_id must be a UUID"):
                await read(str(CLAIM_ID), OWNER, session=self.session)
        self.session.execute.assert_not_called()

    async def test_owned_state_preserves_each_lifecycle_state(self):
        for status in ("queued", "processing", "completed", "failed"):
            with self.subTest(status=status):
                row = {"claim_id": CLAIM_ID, "processing_status": status,
                       "queued_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                       "started_at": None, "completed_at": None, "failed_at": None,
                       "failure_category": "interrupted" if status == "failed" else None}
                self.return_row(row)
                self.assertEqual(await repo.get_owned_claim_state(CLAIM_ID, OWNER, session=self.session), row)
                self.assertEqual(self.compiled().params, {"claim_id_1": CLAIM_ID, "anonymous_session_hash_1": OWNER})
        self.assertEqual(self.session.execute.await_count, 4)
        self.assert_caller_owned_transaction()

    async def test_both_reads_scope_sql_to_claim_and_owner(self):
        self.return_row(None)
        for read in (repo.get_owned_claim_state, repo.get_owned_claim_result):
            for owner in (OWNER, OTHER):
                with self.subTest(read=read.__name__, owner=owner):
                    self.assertIsNone(await read(CLAIM_ID, owner, session=self.session))
                    compiled = self.compiled()
                    where = str(compiled).split("WHERE", 1)[1]
                    self.assertIn("c3.claims.claim_id =", where)
                    self.assertIn("AND c3.claims.anonymous_session_hash =", where)
                    self.assertEqual(compiled.params["anonymous_session_hash_1"], owner)
                    self.assertEqual(compiled.params["claim_id_1"], CLAIM_ID)
        self.assert_caller_owned_transaction()

    async def test_completed_snapshot_is_returned_unchanged(self):
        payload = {"claim_verdict": None, "insufficient_evidence": True,
                   "citations": [], "evidence": {"passages": []}, "extra": [1, 2]}
        row = {"claim_id": CLAIM_ID, "processing_status": "completed", "result_payload": payload}
        self.return_row(row)
        result = await repo.get_owned_claim_result(CLAIM_ID, OWNER, session=self.session)
        self.assertEqual(result, row)
        self.assertIs(result["result_payload"], payload)
        sql = str(self.compiled())
        self.assertIn("LEFT OUTER JOIN c3.verification_results", sql)
        self.assertIn("c3.verification_results.claim_id = c3.claims.claim_id", sql)
        self.session.execute.assert_awaited_once()
        self.assert_caller_owned_transaction()

    async def test_owned_claim_without_result_retains_metadata(self):
        for status in ("queued", "processing", "failed", "completed"):
            row = {"claim_id": CLAIM_ID, "processing_status": status, "result_payload": None}
            self.return_row(row)
            self.assertEqual(await repo.get_owned_claim_result(CLAIM_ID, OWNER, session=self.session), row)

    async def test_errors_are_sanitized_without_ending_callers_transaction(self):
        failure = SQLAlchemyError("private database detail")
        self.session.execute.side_effect = failure
        with self.assertRaisesRegex(repo.ClaimPersistenceError, "^Unable to save the claim\\.$") as caught:
            await repo.create_text_claim("text", anonymous_session_hash=OWNER, session=self.session)
        self.assertIs(caught.exception.__cause__, failure)
        for read in (repo.get_owned_claim_state, repo.get_owned_claim_result):
            with self.assertRaisesRegex(repo.ClaimStorageError, "^Unable to read claim storage\\.$"):
                await read(CLAIM_ID, OWNER, session=self.session)
        self.assert_caller_owned_transaction()


if __name__ == "__main__":
    unittest.main()
