"""Offline SQL/transaction-intent tests, not PostgreSQL concurrency tests."""

import unittest
from unittest.mock import MagicMock, create_autospec
from uuid import UUID

from sqlalchemy.dialects.postgresql import JSONB, dialect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.components.c3_misinformation import repository as repo
from app.components.c3_misinformation.models import VerificationResult


CLAIM_ID = UUID("03030303-0303-4303-8303-030303030303")
TOKEN = UUID("04040404-0404-4404-8404-040404040404")


def payload():
    return {
        "mode": "claim", "response_origin": "application", "claim_verdict": None,
        "explanation": "No evidence", "citations": [], "insufficient_evidence": True,
        "evidence": {"passages": []}, "limitations": [], "validation": {"status": "passed"},
        "model": None, "usage": None, "attempts": 0, "finish_reason": None,
    }


def result(value=None):
    row = MagicMock()
    row.scalar_one_or_none.return_value = value
    row.mappings.return_value.one_or_none.return_value = value
    return row


class WorkerLifecycleRepositoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = create_autospec(AsyncSession, instance=True)
        self.session.in_transaction.return_value = False
        self.transaction = MagicMock()
        self.transaction.__aexit__.return_value = False
        self.session.begin.return_value = self.transaction

    def compiled(self, index):
        return self.session.execute.call_args_list[index].args[0].compile(dialect=dialect())

    def assert_transaction_exit(self, error=None):
        self.session.begin.assert_called_once_with()
        self.transaction.__aenter__.assert_awaited_once()
        self.transaction.__aexit__.assert_awaited_once()
        self.assertIs(self.transaction.__aexit__.call_args.args[0], error)
        self.session.close.assert_not_called()
        self.session.commit.assert_not_called()
        self.session.rollback.assert_not_called()

    def assert_fenced_update(self):
        lock = str(self.compiled(0))
        self.assertIn("FOR UPDATE", lock)
        self.assertNotIn("clock_timestamp", lock)
        update = self.compiled(1)
        where = str(update).split("WHERE", 1)[1]
        self.assertIn("c3.claims.claim_id =", where)
        self.assertIn("c3.claims.worker_token =", where)
        self.assertIn("c3.claims.processing_status =", where)
        self.assertIn("c3.claims.lease_expires_at > clock_timestamp()", where)
        self.assertIn(TOKEN, update.params.values())
        self.assertIn(CLAIM_ID, update.params.values())
        self.assertIn("processing", update.params.values())

    async def test_acquisition_locks_one_ordered_text_claim_and_assigns_lease(self):
        self.session.execute.return_value = result({
            "claim_id": CLAIM_ID, "raw_input_text": "  original  ", "cleaned_text": "original",
        })
        claimed = await repo.claim_next_queued_text_claim(session=self.session, lease_seconds=90)
        self.assertEqual((claimed.claim_id, claimed.raw_input_text, claimed.cleaned_text),
                         (CLAIM_ID, "  original  ", "original"))
        self.assertEqual(claimed.worker_token.version, 4)
        compiled = self.compiled(0)
        sql = str(compiled)
        self.assertIn("FOR UPDATE SKIP LOCKED", sql)
        self.assertIn("ORDER BY c3.claims.queued_at, c3.claims.claim_id", sql)
        self.assertIn("LIMIT", sql)
        self.assertIn("UPDATE c3.claims", sql)
        self.assertIn("FROM next_claim", sql)
        self.assertIn("clock_timestamp() +", sql)
        self.assertIn("INTERVAL '1 second'", sql)
        for value in ("queued", "text", "processing", 1, 90, claimed.worker_token):
            self.assertIn(value, compiled.params.values())
        self.session.execute.assert_awaited_once()
        self.assert_transaction_exit()

    async def test_empty_queue_returns_none(self):
        self.session.execute.return_value = result()
        self.assertIsNone(await repo.claim_next_queued_text_claim(session=self.session))
        self.assert_transaction_exit()

    async def test_renewal_checks_after_row_lock_and_does_not_shorten_deadline(self):
        self.session.execute.side_effect = [result(), result(CLAIM_ID)]
        await repo.renew_claim_lease(CLAIM_ID, TOKEN, session=self.session, lease_seconds=60)
        self.assert_fenced_update()
        self.assertIn("greatest(c3.claims.lease_expires_at, clock_timestamp() +", str(self.compiled(1)))
        self.assertIn(60, self.compiled(1).params.values())
        self.assert_transaction_exit()

    async def test_expired_wrong_token_and_missing_lease_renewals_are_rejected(self):
        # Each condition is represented by PostgreSQL returning no updated row.
        for reason in ("expired", "wrong token", "missing claim"):
            with self.subTest(reason=reason):
                self.setUp()
                self.session.execute.side_effect = [result(), result()]
                with self.assertRaisesRegex(repo.ClaimStateError, "expired or no longer owned"):
                    await repo.renew_claim_lease(CLAIM_ID, TOKEN, session=self.session)
                self.assert_fenced_update()
                self.assert_transaction_exit(repo.ClaimStateError)

    async def test_extraction_write_is_fenced_and_preserves_text(self):
        self.session.execute.side_effect = [result(), result(CLAIM_ID)]
        await repo.record_extracted_claim(CLAIM_ID, "  exact text  ", TOKEN, session=self.session)
        self.assert_fenced_update()
        self.assertEqual(self.compiled(1).params["extracted_claim"], "  exact text  ")
        self.assert_transaction_exit()

    async def test_stale_extraction_completion_and_failure_cannot_write(self):
        operations = (
            lambda: repo.record_extracted_claim(CLAIM_ID, "text", TOKEN, session=self.session),
            lambda: repo.record_completed_result(CLAIM_ID, payload(), retrieval_top_k=5,
                                                worker_token=TOKEN, session=self.session),
            lambda: repo.record_claim_failure(CLAIM_ID, "provider", worker_token=TOKEN, session=self.session),
        )
        for operation in operations:
            self.setUp()
            self.session.execute.side_effect = [result(), result()]
            with self.assertRaises(repo.ClaimStateError):
                await operation()
            self.assertEqual(self.session.execute.await_count, 2)  # No result insert.
            self.assert_fenced_update()
            self.assert_transaction_exit(repo.ClaimStateError)

    async def test_completion_status_and_snapshot_share_one_transaction(self):
        self.session.execute.side_effect = [result(), result(CLAIM_ID), result()]
        original = payload()
        original["extra"] = {"preserved": [1]}
        result_id = await repo.record_completed_result(
            CLAIM_ID, original, retrieval_top_k=5, worker_token=TOKEN, session=self.session,
        )
        self.assertEqual(result_id.version, 4)
        self.assert_fenced_update()
        transition = self.compiled(1).params
        self.assertEqual(transition["processing_status"], "completed")
        self.assertIsNone(transition["worker_token"])
        self.assertIsNone(transition["lease_expires_at"])
        compiled = self.compiled(2)
        self.assertIn("INSERT INTO c3.verification_results", str(compiled))
        self.assertEqual(compiled.params["result_id"], result_id)
        self.assertEqual(compiled.params["verdict"], "insufficient_evidence")
        self.assertIsNone(compiled.params["confidence_score"])
        self.assertEqual(compiled.params["result_payload"], original)
        original["extra"]["preserved"].append(2)
        self.assertEqual(compiled.params["result_payload"]["extra"], {"preserved": [1]})
        self.assertIsInstance(VerificationResult.__table__.c.result_payload.type, JSONB)
        self.assert_transaction_exit()

    async def test_supported_snapshot_verdict_is_preserved(self):
        self.session.execute.side_effect = [result(), result(CLAIM_ID), result()]
        supported = payload()
        supported.update(claim_verdict="False", insufficient_evidence=False, model="fixture")
        await repo.record_completed_result(CLAIM_ID, supported, retrieval_top_k=50,
                                          worker_token=TOKEN, session=self.session)
        params = self.compiled(2).params
        self.assertEqual(params["verdict"], "False")
        self.assertEqual(params["llm_model_name"], "fixture")
        self.assertEqual(params["result_payload"], supported)

    async def test_duplicate_insert_exits_transaction_with_error_for_rollback(self):
        failure = IntegrityError("insert", {}, RuntimeError("duplicate claim result"))
        self.session.execute.side_effect = [result(), result(CLAIM_ID), failure]
        with self.assertRaisesRegex(repo.ClaimStorageError, "^Unable to update claim storage\\.$") as caught:
            await repo.record_completed_result(CLAIM_ID, payload(), retrieval_top_k=5,
                                              worker_token=TOKEN, session=self.session)
        self.assertIs(caught.exception.__cause__, failure)
        self.assertTrue(VerificationResult.__table__.c.claim_id.unique)
        self.assert_transaction_exit(IntegrityError)

    async def test_failure_clears_lease_and_stores_only_safe_category(self):
        self.session.execute.side_effect = [result(), result(CLAIM_ID)]
        await repo.record_claim_failure(CLAIM_ID, "provider", worker_token=TOKEN, session=self.session)
        self.assert_fenced_update()
        params = self.compiled(1).params
        self.assertEqual(params["processing_status"], "failed")
        self.assertEqual(params["failure_category"], "provider")
        self.assertIsNone(params["worker_token"])
        self.assertIsNone(params["lease_expires_at"])
        self.assert_transaction_exit()

    async def test_recovery_marks_only_expired_leased_processing_work_interrupted(self):
        rows = result()
        rows.scalars.return_value = [CLAIM_ID]
        self.session.execute.return_value = rows
        self.assertEqual(await repo.recover_expired_claims(session=self.session), [CLAIM_ID])
        compiled = self.compiled(0)
        self.assertIn("worker_token IS NOT NULL", str(compiled))
        self.assertIn("lease_expires_at <= clock_timestamp()", str(compiled))
        self.assertIn("processing", compiled.params.values())
        self.assertEqual(compiled.params["failure_category"], "interrupted")
        self.assertEqual(compiled.params["processing_status"], "failed")
        self.assertNotIn("queued", compiled.params.values())
        self.assertIsNone(compiled.params["worker_token"])
        self.assertIsNone(compiled.params["lease_expires_at"])
        self.assert_transaction_exit()

    async def test_validation_happens_before_transaction_or_sql(self):
        for seconds in (True, 0, 86401, 1.0):
            with self.assertRaises(ValueError):
                await repo.claim_next_queued_text_claim(session=self.session, lease_seconds=seconds)
            with self.assertRaises(ValueError):
                await repo.renew_claim_lease(CLAIM_ID, TOKEN, session=self.session, lease_seconds=seconds)
        with self.assertRaises(ValueError):
            await repo.record_extracted_claim(CLAIM_ID, " ", TOKEN, session=self.session)
        with self.assertRaises(ValueError):
            await repo.record_completed_result(CLAIM_ID, {}, retrieval_top_k=5,
                                              worker_token=TOKEN, session=self.session)
        with self.assertRaises(ValueError):
            await repo.record_completed_result(CLAIM_ID, payload(), retrieval_top_k=True,
                                              worker_token=TOKEN, session=self.session)
        with self.assertRaises(ValueError):
            await repo.record_claim_failure(CLAIM_ID, "private error detail", worker_token=TOKEN, session=self.session)
        with self.assertRaises(ValueError):
            await repo.record_claim_failure(CLAIM_ID, "provider", worker_token=None, session=self.session)
        self.session.execute.assert_not_called()
        self.session.begin.assert_not_called()

    async def test_active_caller_transaction_is_rejected_without_commit_or_rollback(self):
        self.session.in_transaction.return_value = True
        with self.assertRaisesRegex(ValueError, "require an idle session"):
            await repo.claim_next_queued_text_claim(session=self.session)
        self.session.begin.assert_not_called()
        self.session.execute.assert_not_called()
        self.session.commit.assert_not_called()
        self.session.rollback.assert_not_called()

    async def test_transaction_exit_failure_does_not_report_success(self):
        self.session.execute.return_value = result()
        self.transaction.__aexit__.side_effect = RuntimeError("commit failed")
        with self.assertRaises(repo.ClaimStorageError):
            await repo.claim_next_queued_text_claim(session=self.session)


if __name__ == "__main__":
    unittest.main()
