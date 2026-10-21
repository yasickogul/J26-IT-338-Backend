"""Offline vertical-slice tests for the C3 claim lifecycle."""

import asyncio
import json
import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.components.c3_misinformation import (
    api,
    claim_worker_runner,
    repository,
    worker_dependencies,
)
from app.components.c3_misinformation.agents.gemini_adapter import (
    DEFAULT_MODEL,
    GeminiAdapterError,
    GeminiGeneration,
)
from app.components.c3_misinformation.anonymous_claim_session import (
    COOKIE_NAME,
    SECURE_COOKIE_NAME,
    session_token_hash,
)
from app.components.c3_misinformation.claim_lifecycle_contract import (
    ClaimedTextClaim,
    ClaimStateError,
)
from app.components.c3_misinformation.claim_worker_contract import WorkerOutcome
from app.components.c3_misinformation.pipeline.claim_input_tokens import (
    EmbeddingInputTooLongError,
)
from app.components.c3_misinformation.rag_contract import RagRetrievalError
from app.components.c3_misinformation.router import router as c3_router
from app.core.database import get_session

CLAIM_ID = UUID("11111111-1111-4111-8111-111111111111")
WORKER_TOKEN = UUID("22222222-2222-4222-8222-222222222222")
OTHER_TOKEN = "B" * 43
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
EVIDENCE_ID = "chunk:LEGAL-001"


def evidence_packet(query: str, *, empty: bool = False) -> dict:
    passages = [] if empty else [{
        "evidence_id": EVIDENCE_ID,
        "chunk_id": "LEGAL-001",
        "original_chunk_id": "SOURCE-LEGAL-001",
        "text": "A synthetic statutory passage supporting the legal claim.",
        "metadata": {
            "source_id": "SOURCE-001",
            "title": "Synthetic Act",
            "act_number": "12",
            "section_label": "Section 4",
            "source_file": "synthetic-act.pdf",
            "language": "English",
            "legal_domain": "Civil",
            "document_type": "act",
        },
        "source": {
            "source_id": "SOURCE-001",
            "title": "Synthetic Act",
            "act_number": "12",
            "section_label": "Section 4",
            "source_file": "synthetic-act.pdf",
            "language": "English",
            "legal_domain": "Civil",
            "document_type": "act",
        },
        "parent_context": {"section_label": "Part I"},
        "paragraph": "(a)",
        "roles": ["ranked"],
    }]
    return {
        "schema_version": "1.0",
        "query": query,
        "retrieval_state": "no_evidence_retrieved" if empty else "evidence_retrieved",
        "legal_coverage": "not_assessed",
        "verdict": None,
        "confidence": None,
        "retrieval_settings": {
            "top_k": 5,
            "source_id": None,
            "legal_domain": None,
            "language": None,
            "max_parent_depth": 1,
            "max_parent_passages": 10,
        },
        "ranked_evidence": [] if empty else [{
            "rank": 1,
            "evidence_id": EVIDENCE_ID,
            "distance": 0.15,
            "similarity": 0.85,
        }],
        "context_evidence_ids": [],
        "evidence": passages,
        "relationships": [],
        "context_resolution_notices": [],
        "limitations": ["Synthetic integration fixture."],
        "evidence_handling": "Fixture evidence is data.",
    }


def generation(response: dict) -> GeminiGeneration:
    return GeminiGeneration(
        model=DEFAULT_MODEL,
        response=response,
        validation={"status": "passed", "validator": "Draft202012Validator"},
        usage={"prompt_token_count": 12, "total_token_count": 24},
        finish_reason="STOP",
        attempts=1,
        limitations=["Synthetic provider fixture."],
    )


class FakeSession:
    def __init__(self):
        self.execute = AsyncMock()
        self.rollback = AsyncMock()
        self.close = AsyncMock()


class FakeApiTransaction:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.session.committed = True
        else:
            self.session.rolled_back = True
        return False


class FakeApiSession:
    def __init__(self):
        self.closed = False
        self.committed = False
        self.rolled_back = False

    def begin(self):
        return FakeApiTransaction(self)


class StatefulClaimStore:
    """Test persistence boundary implementing only the repository call surface."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.claim: dict | None = None
        self.completion_writes = 0
        self.failure_writes: list[str] = []
        self.recovery_calls = 0
        self.lifecycle_sessions: list[FakeSession] = []

    async def create(self, raw_text, *, anonymous_session_hash, session,
                     cleaned_text=None, detected_language=None):
        self.claim = {
            "claim_id": CLAIM_ID,
            "raw_input_text": raw_text,
            "cleaned_text": cleaned_text,
            "anonymous_session_hash": anonymous_session_hash,
            "detected_language": detected_language,
            "processing_status": "queued",
            "queued_at": NOW,
            "started_at": None,
            "completed_at": None,
            "failed_at": None,
            "failure_category": None,
            "result_payload": None,
        }
        return CLAIM_ID

    async def session_has_claim(self, owner_hash, *, session):
        return bool(self.claim and self.claim["anonymous_session_hash"] == owner_hash)

    async def owned_result(self, claim_id, owner_hash, *, session):
        if (
            self.claim is None
            or claim_id != self.claim["claim_id"]
            or owner_hash != self.claim["anonymous_session_hash"]
        ):
            return None
        return dict(self.claim)

    async def recover(self, *, session):
        self.recovery_calls += 1
        return []

    async def acquire(self, *, session, lease_seconds=3600):
        if self.claim is None or self.claim["processing_status"] != "queued":
            return None
        self.claim["processing_status"] = "processing"
        self.claim["started_at"] = NOW
        self.claim["worker_token"] = WORKER_TOKEN
        self.claim["lease_expires_at"] = NOW
        return ClaimedTextClaim(
            claim_id=CLAIM_ID,
            raw_input_text=self.claim["raw_input_text"],
            cleaned_text=self.claim["cleaned_text"],
            worker_token=WORKER_TOKEN,
        )

    async def renew(self, claim_id, worker_token, *, session, lease_seconds=3600):
        if not self._owns_lease(claim_id, worker_token):
            raise ClaimStateError("Claim is not processing under this worker lease.")

    async def complete(self, claim_id, result_payload, *, retrieval_top_k,
                       worker_token, session):
        if not self._owns_lease(claim_id, worker_token):
            raise ClaimStateError("Claim is not processing under this worker lease.")
        if self.claim["result_payload"] is not None:
            raise AssertionError("duplicate completion write")
        self.completion_writes += 1
        self.claim.update(
            processing_status="completed",
            completed_at=NOW,
            failure_category=None,
            result_payload=result_payload,
            retrieval_top_k=retrieval_top_k,
            worker_token=None,
            lease_expires_at=None,
        )
        return uuid4()

    async def fail(self, claim_id, category, *, worker_token, session):
        if not self._owns_lease(claim_id, worker_token):
            raise ClaimStateError("Claim is not processing under this worker lease.")
        self.failure_writes.append(category)
        self.claim.update(
            processing_status="failed",
            failed_at=NOW,
            failure_category=category,
            worker_token=None,
            lease_expires_at=None,
        )

    def _owns_lease(self, claim_id, worker_token):
        return (
            self.claim is not None
            and self.claim["claim_id"] == claim_id
            and self.claim["processing_status"] == "processing"
            and self.claim.get("worker_token") == worker_token
        )

    def session_factory(self):
        session = FakeSession()
        self.lifecycle_sessions.append(session)
        return session


class FakeSyncGeminiAdapter:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def generate_json(self, prompt, schema):
        self.calls.append((prompt, schema))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class ClaimLifecycleIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = StatefulClaimStore()
        self.api_sessions: list[FakeApiSession] = []
        self.app = FastAPI()
        self.app.include_router(c3_router, prefix="/api/v1/c3")

        async def api_session_dependency():
            session = FakeApiSession()
            self.api_sessions.append(session)
            try:
                yield session
            finally:
                session.closed = True

        self.app.dependency_overrides[get_session] = api_session_dependency
        self.app.dependency_overrides[api.get_claim_token_validator] = lambda: (
            lambda texts: [len(text) for text in texts]
        )
        self.client = TestClient(self.app)
        self.patchers = [
            patch.object(repository, "create_text_claim", side_effect=self.store.create),
            patch.object(
                repository,
                "anonymous_session_has_claim",
                side_effect=self.store.session_has_claim,
            ),
            patch.object(
                repository,
                "get_owned_claim_result",
                side_effect=self.store.owned_result,
            ),
            patch.object(repository, "recover_expired_claims", side_effect=self.store.recover),
            patch.object(repository, "claim_next_queued_text_claim", side_effect=self.store.acquire),
            patch.object(repository, "renew_claim_lease", side_effect=self.store.renew),
            patch.object(repository, "record_completed_result", side_effect=self.store.complete),
            patch.object(repository, "record_claim_failure", side_effect=self.store.fail),
        ]
        self.repository_mocks = [patcher.start() for patcher in self.patchers]

    def tearDown(self):
        self.client.close()
        self.app.dependency_overrides.clear()
        for patcher in reversed(self.patchers):
            patcher.stop()

    def make_dependencies(self, *, packet=None, provider_outcome=None, packet_error=None):
        async def build_evidence(query, **_filters):
            if packet_error is not None:
                raise packet_error
            return packet if packet is not None else evidence_packet(query)

        adapter = FakeSyncGeminiAdapter(
            provider_outcome
            if provider_outcome is not None
            else generation({
                "claim_verdict": "False",
                "citations": [EVIDENCE_ID],
                "insufficient_evidence": False,
                "explanation": "The fixture statute contradicts the submitted claim.",
            })
        )
        dependencies = worker_dependencies.create_worker_dependencies(
            session_factory=self.store.session_factory,
            query_embedder=AsyncMock(),
            gemini_adapter_factory=lambda: adapter,
            token_validator=lambda texts: [len(text) for text in texts],
        )
        return dependencies, adapter, patch.object(
            worker_dependencies,
            "build_c3_evidence_packet",
            new=AsyncMock(side_effect=build_evidence),
        )

    def submit(self, text="The statute says no legal claim.", *, client=None):
        return (client or self.client).post(
            "/api/v1/c3/claims/analyze",
            json={"text": text},
        )

    async def test_full_submit_worker_rag_serialize_and_owner_result_flow(self):
        dependencies, adapter, evidence_patch = self.make_dependencies()
        with evidence_patch:
            submitted = self.submit()

            self.assertEqual(submitted.status_code, 202)
            self.assertEqual(submitted.json(), {
                "claim_id": str(CLAIM_ID),
                "status": "processing",
            })
            self.assertEqual(self.store.claim["processing_status"], "queued")
            self.assertTrue(self.api_sessions[-1].committed)
            self.assertEqual(self.store.claim["anonymous_session_hash"],
                             session_token_hash(submitted.cookies[COOKIE_NAME]))
            self.assertNotIn(submitted.cookies[COOKIE_NAME],
                             self.store.claim["anonymous_session_hash"])
            self.assertEqual(self.store.completion_writes, 0)
            self.assertEqual(adapter.calls, [])

            queued = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
            self.assertEqual(queued.status_code, 200)
            self.assertEqual(queued.json()["status"], "queued")
            self.assertIsNone(queued.json()["result"])
            self.assertEqual(self.store.claim["processing_status"], "queued")
            self.assertEqual(adapter.calls, [])

            outcome = await claim_worker_runner.run_configured_worker(
                dependencies=dependencies,
                config=claim_worker_runner.WorkerRunnerConfig(lease_seconds=60),
            )

            self.assertEqual(outcome, WorkerOutcome(CLAIM_ID, "completed"))
            self.assertEqual(self.store.claim["processing_status"], "completed")
            self.assertEqual(self.store.completion_writes, 1)
            self.assertEqual(self.store.claim["retrieval_top_k"], 5)
            self.assertEqual(len(adapter.calls), 1)
            prompt, schema = adapter.calls[0]
            self.assertEqual(prompt["input"], {
                "mode": "claim", "claim": "The statute says no legal claim.",
            })
            self.assertEqual(prompt["evidence_data"]["passages"][0]["text"],
                             "A synthetic statutory passage supporting the legal claim.")
            self.assertEqual(
                schema["properties"]["citations"]["items"]["enum"],
                [EVIDENCE_ID],
            )

            completed = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
            self.assertEqual(completed.status_code, 200)
            self.assertEqual(completed.headers["cache-control"], "no-store")
            body = completed.json()
            self.assertEqual(body["status"], "completed")
            result = body["result"]
            self.assertEqual(result["mode"], "claim")
            self.assertEqual(result["claim_verdict"], "False")
            self.assertEqual(result["citations"], [EVIDENCE_ID])
            self.assertEqual(result["evidence"]["passages"][0]["evidence_id"], EVIDENCE_ID)
            self.assertEqual(result["evidence"]["passages"][0]["original_chunk_id"],
                             "SOURCE-LEGAL-001")
            self.assertEqual(result["evidence"]["passages"][0]["metadata"]["title"],
                             "Synthetic Act")
            self.assertEqual(result["evidence"]["passages"][0]["source"]["source_id"],
                             "SOURCE-001")
            self.assertEqual(result["evidence"]["passages"][0]["text"],
                             "A synthetic statutory passage supporting the legal claim.")
            self.assertEqual(result["evidence"]["ranked_evidence"][0]["distance"], 0.15)
            self.assertEqual(result["evidence"]["ranked_evidence"][0]["similarity"], 0.85)
            self.assertEqual(result["evidence"]["relationships"], [])
            self.assertEqual(result["evidence"]["context_resolution_notices"], [])
            json.dumps(result, allow_nan=False)
            self.assertEqual(self.store.claim["result_payload"], result)
            self.assertEqual(len(self.store.lifecycle_sessions), 4)
            self.assertTrue(all(session.close.await_count == 1
                                for session in self.store.lifecycle_sessions))
            self.assertTrue(all(session.rollback.await_count == 1
                                for session in self.store.lifecycle_sessions
                                if session.execute.await_count))
            self.assertTrue(all(session.closed for session in self.api_sessions))

            no_duplicate = await claim_worker_runner.run_configured_worker(
                dependencies=dependencies,
                config=claim_worker_runner.WorkerRunnerConfig(lease_seconds=60),
            )
            self.assertIsNone(no_duplicate)
            self.assertEqual(self.store.completion_writes, 1)

        openapi = self.client.get("/openapi.json").json()
        paths = openapi["paths"]
        self.assertIn("/api/v1/c3/claims/analyze", paths)
        self.assertIn("/api/v1/c3/results/{claim_id}", paths)
        request_schema = paths["/api/v1/c3/claims/analyze"]["post"]["requestBody"]
        schema_ref = request_schema["content"]["application/json"]["schema"]["$ref"]
        self.assertIn(
            "text",
            openapi["components"]["schemas"][schema_ref.rsplit("/", 1)[-1]]["properties"],
        )
        self.assertEqual(
            paths["/api/v1/c3/claims/analyze"]["post"]["responses"]["202"]["description"],
            "Successful Response",
        )

    async def test_empty_evidence_abstains_without_provider_and_preserves_result(self):
        dependencies, adapter, evidence_patch = self.make_dependencies(
            packet=evidence_packet("The statute says no legal claim.", empty=True),
        )
        with evidence_patch:
            submitted = self.submit()
            outcome = await claim_worker_runner.run_configured_worker(
                dependencies=dependencies,
                config=claim_worker_runner.WorkerRunnerConfig(lease_seconds=60),
            )
            result = self.client.get(
                f"/api/v1/c3/results/{CLAIM_ID}",
                cookies={COOKIE_NAME: submitted.cookies[COOKIE_NAME]},
            )
        self.assertEqual(submitted.status_code, 202)
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(result.status_code, 200)
        payload = result.json()["result"]
        self.assertTrue(payload["insufficient_evidence"])
        self.assertIsNone(payload["claim_verdict"])
        self.assertEqual(payload["citations"], [])
        self.assertEqual(payload["evidence"]["passages"], [])
        self.assertEqual(adapter.calls, [])
        self.assertEqual(self.store.completion_writes, 1)

    def test_input_and_cookie_contracts_and_no_http_worker_start(self):
        invalid = self.client.post("/api/v1/c3/claims/analyze", json={"text": " "})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.json()["detail"]["code"], "invalid_claim_text")

        malformed = self.client.post(
            "/api/v1/c3/claims/analyze",
            json={"text": "valid claim", "extra": True},
        )
        self.assertEqual(malformed.status_code, 422)

        def reject_long_input(_texts):
            raise EmbeddingInputTooLongError(index=0, token_count=4, limit=3)

        self.app.dependency_overrides[api.get_claim_token_validator] = lambda: reject_long_input
        too_long = self.submit()
        self.assertEqual(too_long.status_code, 400)
        self.assertEqual(too_long.json()["detail"]["limit"], 3)

        self.app.dependency_overrides[api.get_claim_token_validator] = lambda: (
            lambda texts: [len(text) for text in texts]
        )
        owner = self.submit()
        self.assertEqual(owner.status_code, 202)
        self.assertIn("httponly", owner.headers["set-cookie"].lower())
        self.assertIn("samesite=lax", owner.headers["set-cookie"].lower())
        self.assertNotIn("secure", owner.headers["set-cookie"].lower())
        self.assertIn("processing", owner.json()["status"])
        self.assertEqual(self.store.claim["processing_status"], "queued")

        reused = self.client.post(
            "/api/v1/c3/claims/analyze",
            json={"text": "another claim"},
        )
        self.assertEqual(reused.status_code, 202)
        self.assertNotIn("set-cookie", reused.headers)
        self.assertEqual(self.store.claim["anonymous_session_hash"],
                         session_token_hash(owner.cookies[COOKIE_NAME]))

        https_client = TestClient(self.app, base_url="https://testserver")
        try:
            secure = https_client.post(
                "/api/v1/c3/claims/analyze",
                json={"text": "secure cookie fixture"},
            )
        finally:
            https_client.close()
        self.assertEqual(secure.status_code, 202)
        cookie_header = secure.headers["set-cookie"].lower()
        self.assertTrue(cookie_header.startswith(SECURE_COOKIE_NAME.lower() + "="))
        self.assertIn("secure", cookie_header)
        self.assertIn("httponly", cookie_header)
        self.assertIn("samesite=lax", cookie_header)
        self.assertNotIn("domain=", cookie_header)
        self.assertEqual(self.store.completion_writes, 0)

    async def test_all_lifecycle_states_owner_isolation_and_non_disclosure(self):
        owner = self.submit()
        owner_token = owner.cookies[COOKIE_NAME]
        foreign_client = TestClient(self.app)
        try:
            foreign_client.cookies.set(COOKIE_NAME, OTHER_TOKEN)
            for state in ("queued", "processing", "failed"):
                self.store.claim["processing_status"] = state
                self.store.claim["result_payload"] = None
                response = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], state)
                self.assertIsNone(response.json()["result"])
                self.assertNotIn("failure_category", response.json())

            unauthenticated = TestClient(self.app)
            try:
                no_cookie = unauthenticated.get(f"/api/v1/c3/results/{CLAIM_ID}")
            finally:
                unauthenticated.close()
            self.assertEqual(no_cookie.status_code, 401)

            foreign = foreign_client.get(f"/api/v1/c3/results/{CLAIM_ID}")
            self.assertEqual(foreign.status_code, 404)
            unknown = self.client.get(f"/api/v1/c3/results/{uuid4()}")
            self.assertEqual(unknown.status_code, 404)
            self.assertEqual(foreign.json(), unknown.json())
            self.assertEqual(foreign.headers["cache-control"], "no-store")
            self.assertEqual(unknown.headers["cache-control"], "no-store")

            self.store.claim["processing_status"] = "completed"
            self.store.claim["result_payload"] = {"preserve": {"any": ["stored", 1]}}
            completed = self.client.get(
                f"/api/v1/c3/results/{CLAIM_ID}",
                cookies={COOKIE_NAME: owner_token},
            )
            self.assertEqual(completed.status_code, 200)
            self.assertEqual(completed.json()["result"], {"preserve": {"any": ["stored", 1]}})

            invalid_client = TestClient(self.app)
            try:
                # Keep this request unambiguous: self.client still owns the
                # valid cookie, and httpx can retain both values when a cookie
                # is reset with different domain/path metadata.
                invalid_client.cookies.set(COOKIE_NAME, "invalid")
                invalid_session = invalid_client.get(
                    f"/api/v1/c3/results/{CLAIM_ID}",
                )
            finally:
                invalid_client.close()
            self.assertEqual(invalid_session.status_code, 401)
        finally:
            foreign_client.close()

    async def test_provider_retrieval_and_citation_failures_are_stored_as_safe_failures(self):
        cases = (
            (
                generation({
                    "claim_verdict": "False",
                    "citations": ["not-in-evidence"],
                    "insufficient_evidence": False,
                    "explanation": "Unsupported citation.",
                }),
                None,
                "response_validation",
            ),
            (
                GeminiAdapterError("private provider detail", failure_category="provider_error"),
                None,
                "service",
            ),
            (
                None,
                RagRetrievalError("private retrieval detail", category="retrieval"),
                "retrieval",
            ),
        )
        for provider_outcome, packet_error, expected_category in cases:
            with self.subTest(category=expected_category):
                self.store.reset()
                dependencies, _adapter, evidence_patch = self.make_dependencies(
                    provider_outcome=provider_outcome,
                    packet_error=packet_error,
                )
                self.submit()
                with evidence_patch:
                    outcome = await claim_worker_runner.run_configured_worker(
                        dependencies=dependencies,
                        config=claim_worker_runner.WorkerRunnerConfig(lease_seconds=60),
                    )
                self.assertEqual(outcome.status, "failed")
                self.assertEqual(self.store.claim["processing_status"], "failed")
                self.assertEqual(self.store.failure_writes, [expected_category])
                self.assertIsNone(self.store.claim["result_payload"])
                failed_result = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
                self.assertEqual(failed_result.status_code, 200)
                self.assertEqual(failed_result.json()["status"], "failed")
                self.assertNotIn("private", failed_result.text)
                self.assertNotIn("failure_category", failed_result.json())

    async def test_worker_ownership_loss_cancellation_and_empty_queue(self):
        dependencies, _adapter, evidence_patch = self.make_dependencies()
        self.submit()
        self.store.claim["processing_status"] = "processing"
        self.store.claim["worker_token"] = WORKER_TOKEN
        self.store.claim["result_payload"] = None
        with patch.object(
            repository,
            "record_completed_result",
            new=AsyncMock(side_effect=ClaimStateError("stale worker")),
        ), patch.object(
            repository,
            "claim_next_queued_text_claim",
            new=AsyncMock(return_value=ClaimedTextClaim(
                CLAIM_ID,
                self.store.claim["raw_input_text"],
                WORKER_TOKEN,
                self.store.claim["cleaned_text"],
            )),
        ), evidence_patch:
            outcome = await claim_worker_runner.run_configured_worker(
                dependencies=dependencies,
                config=claim_worker_runner.WorkerRunnerConfig(lease_seconds=60),
            )
        self.assertEqual(outcome.status, "ownership_lost")
        self.assertEqual(self.store.failure_writes, [])

        self.store.reset()
        self.store.acquire = AsyncMock(return_value=None)
        with patch.object(repository, "claim_next_queued_text_claim", new=self.store.acquire):
            no_claim = await claim_worker_runner.run_configured_worker(
                dependencies=dependencies,
                config=claim_worker_runner.WorkerRunnerConfig(lease_seconds=60),
            )
        self.assertIsNone(no_claim)

        self.store.reset()
        self.submit()
        dependencies, _adapter, evidence_patch = self.make_dependencies()
        entered = asyncio.Event()

        async def blocked_evidence(query, **_filters):
            entered.set()
            await asyncio.Event().wait()

        with patch.object(
            worker_dependencies,
            "build_c3_evidence_packet",
            new=AsyncMock(side_effect=blocked_evidence),
        ):
            task = asyncio.create_task(claim_worker_runner.run_configured_worker(
                dependencies=dependencies,
                config=claim_worker_runner.WorkerRunnerConfig(lease_seconds=60),
            ))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.store.completion_writes, 0)
        self.assertEqual(self.store.failure_writes, [])
        self.assertTrue(all(session.close.await_count == 1
                            for session in self.store.lifecycle_sessions))

    async def test_submission_storage_failure_and_openapi_contract(self):
        self.app.dependency_overrides[api.get_claim_token_validator] = lambda: (
            lambda texts: [len(text) for text in texts]
        )
        with patch.object(
            repository,
            "create_text_claim",
            new=AsyncMock(side_effect=RuntimeError("private database details")),
        ):
            failure = self.submit()
        self.assertEqual(failure.status_code, 500)
        self.assertNotIn("private database details", failure.text)
        self.assertNotIn("set-cookie", failure.headers)
        self.assertTrue(self.api_sessions[-1].rolled_back)

        openapi = self.client.get("/openapi.json").json()
        submission = openapi["paths"]["/api/v1/c3/claims/analyze"]["post"]
        self.assertEqual(submission["responses"]["202"]["description"], "Successful Response")
        schema_ref = submission["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        self.assertEqual(schema_ref.rsplit("/", 1)[-1], "ClaimSubmissionRequest")
        self.assertIn("/api/v1/c3/results/{claim_id}", openapi["paths"])


if __name__ == "__main__":
    unittest.main()
