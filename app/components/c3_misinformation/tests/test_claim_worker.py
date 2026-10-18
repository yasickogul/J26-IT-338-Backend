import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from app.components.c3_misinformation import claim_worker as worker
from app.components.c3_misinformation.claim_lifecycle_contract import ClaimStateError
from app.components.c3_misinformation.claim_worker_contract import WorkerOutcome
from app.components.c3_misinformation.rag_contract import RagProviderError, RagResult
from app.components.c3_misinformation.pipeline.claim_extraction import (
    ClaimExtractionProviderResponse,
)
from app.components.c3_misinformation.claim_lifecycle_contract import ClaimedTextClaim


CLAIM_ID = UUID("11111111-1111-4111-8111-111111111111")
TOKEN = UUID("22222222-2222-4222-8222-222222222222")


class Session:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class SessionFactory:
    def __init__(self):
        self.sessions = []

    def __call__(self):
        value = Session()
        self.sessions.append(value)
        return value


def claim(cleaned="cleaned", raw="raw"):
    return ClaimedTextClaim(CLAIM_ID, raw, TOKEN, cleaned)


def rag_result():
    return RagResult(
        mode="claim",
        evidence_packet={
            "ranked_evidence": [], "context_evidence_ids": [], "evidence": [],
            "relationships": [], "context_resolution_notices": [],
        },
        response={
            "claim_verdict": None, "citations": [], "insufficient_evidence": True,
            "explanation": "No evidence was retrieved for this request.",
        },
        response_origin="application", validation={"status": "passed"}, model=None,
        usage=None, attempts=0, finish_reason=None, limitations=[], transaction_status={},
    )


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.factory = SessionFactory()
        self.recover = patch.object(worker.repository, "recover_expired_claims", new=AsyncMock(return_value=[]))
        self.acquire = patch.object(worker.repository, "claim_next_queued_text_claim", new=AsyncMock(return_value=claim()))
        self.complete = patch.object(worker.repository, "record_completed_result", new=AsyncMock())
        self.failure = patch.object(worker.repository, "record_claim_failure", new=AsyncMock())
        self.renew = patch.object(worker.repository, "renew_claim_lease", new=AsyncMock())
        self.extracted = patch.object(worker.repository, "record_extracted_claim", new=AsyncMock())
        self.mocks = [p.start() for p in (self.recover, self.acquire, self.complete, self.failure, self.renew, self.extracted)]
        self.addCleanup(lambda: [p.stop() for p in (self.recover, self.acquire, self.complete, self.failure, self.renew, self.extracted)])

    async def analyzer(self, text, *, top_k):
        self.analysis_text = text
        self.analysis_top_k = top_k
        return rag_result()

    async def test_no_claim_returns_none_and_closes_recovery_and_acquisition_sessions(self):
        self.acquire.stop()
        self.acquire = patch.object(worker.repository, "claim_next_queued_text_claim", new=AsyncMock(return_value=None))
        self.acquire.start()
        result = await worker.process_one_claim(analysis=self.analyzer, session_factory=self.factory)
        self.assertIsNone(result)
        self.assertEqual(len(self.factory.sessions), 2)
        self.assertTrue(all(s.closed for s in self.factory.sessions))

    async def test_success_uses_cleaned_text_and_persists_serialized_result(self):
        result = await worker.process_one_claim(analysis=self.analyzer, session_factory=self.factory, top_k=7)
        self.assertEqual(result, WorkerOutcome(CLAIM_ID, "completed"))
        self.assertEqual(self.analysis_text, "cleaned")
        self.assertEqual(self.analysis_top_k, 7)
        self.mocks[2].assert_awaited_once()
        payload = self.mocks[2].await_args.args[1]
        self.assertTrue(payload["insufficient_evidence"])
        self.assertEqual(self.mocks[2].await_args.kwargs["retrieval_top_k"], 7)

    async def test_raw_text_is_used_when_cleaned_text_is_none(self):
        self.acquire.stop()
        self.acquire = patch.object(worker.repository, "claim_next_queued_text_claim", new=AsyncMock(return_value=claim(cleaned=None)))
        self.acquire.start()
        await worker.process_one_claim(analysis=self.analyzer, session_factory=self.factory)
        self.assertEqual(self.analysis_text, "raw")

    async def test_extraction_is_disabled_by_default(self):
        with patch.dict(os.environ, {"COMPNET_CLAIM_EXTRACTION_ENABLED": "false"}):
            await worker.process_one_claim(analysis=self.analyzer, session_factory=self.factory)
        self.mocks[5].assert_not_awaited()

    async def test_optional_extraction_records_provenance_and_selected_claim(self):
        class Extractor:
            def extract(self, text):
                return ClaimExtractionProviderResponse("extracted", "assertion", "mock")

        await worker.process_one_claim(
            analysis=self.analyzer, session_factory=self.factory,
            extraction_enabled=True, extractor_factory=Extractor,
        )
        self.assertEqual(self.analysis_text, "assertion")
        self.mocks[5].assert_awaited_once()
        payload = self.mocks[2].await_args.args[1]
        self.assertEqual(payload["claim_extraction"]["method"], "mock")

    async def test_extraction_failure_maps_to_input_and_writes_failure(self):
        class Extractor:
            def extract(self, text):
                return ClaimExtractionProviderResponse("multiple_claims", None, "mock")

        result = await worker.process_one_claim(
            analysis=self.analyzer, session_factory=self.factory,
            extraction_enabled=True, extractor_factory=Extractor,
        )
        self.assertEqual(result.status, "failed")
        self.mocks[3].assert_awaited_once()
        self.assertEqual(self.mocks[3].await_args.args[1], "input")

    async def test_rag_category_is_preserved(self):
        async def failing(text, *, top_k):
            raise RagProviderError("provider", category="provider")

        result = await worker.process_one_claim(analysis=failing, session_factory=self.factory)
        self.assertEqual(result, WorkerOutcome(CLAIM_ID, "failed"))
        self.assertEqual(self.mocks[3].await_args.args[1], "provider")

    async def test_ownership_loss_returns_fenced_outcome(self):
        self.mocks[2].side_effect = ClaimStateError("lost")
        result = await worker.process_one_claim(analysis=self.analyzer, session_factory=self.factory)
        self.assertEqual(result, WorkerOutcome(CLAIM_ID, "ownership_lost"))
        self.assertGreaterEqual(self.mocks[0].await_count, 2)
        self.mocks[3].assert_not_awaited()

    async def test_transient_renewal_failure_does_not_stop_processing(self):
        self.mocks[4].side_effect = RuntimeError("temporary")
        # A short lease makes the renewal interval small without real waiting
        # in the analyzer; the operation still succeeds and cleanup is tested.
        result = await worker.process_one_claim(analysis=self.analyzer, session_factory=self.factory, lease_seconds=1)
        self.assertEqual(result.status, "completed")

    async def test_long_analysis_renews_active_lease(self):
        async def slow(text, *, top_k):
            await asyncio.sleep(0.4)
            return rag_result()

        result = await worker.process_one_claim(
            analysis=slow, session_factory=self.factory, lease_seconds=1,
        )
        self.assertEqual(result.status, "completed")
        self.assertGreaterEqual(self.mocks[4].await_count, 1)

    async def test_cancellation_cleans_renewal_and_propagates(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking(text, *, top_k):
            started.set()
            await release.wait()
            return rag_result()

        task = asyncio.create_task(worker.process_one_claim(analysis=blocking, session_factory=self.factory, lease_seconds=1))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.mocks[2].assert_not_awaited()
        self.mocks[3].assert_not_awaited()

    async def test_invalid_arguments_are_rejected_before_recovery(self):
        with self.assertRaises(ValueError):
            await worker.process_one_claim(analysis=self.analyzer, session_factory=self.factory, top_k=0)
        with self.assertRaises(TypeError):
            await worker.process_one_claim(analysis=self.analyzer, session_factory=self.factory, extraction_enabled=1)
        self.mocks[0].assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
