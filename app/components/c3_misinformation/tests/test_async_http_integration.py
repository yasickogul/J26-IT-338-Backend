"""Offline async HTTP integration tests for the real C3 FastAPI routes."""

from datetime import datetime, timezone
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from fastapi import FastAPI
import httpx

from app.components.c3_misinformation import api
from app.components.c3_misinformation.anonymous_claim_session import COOKIE_NAME
from app.components.c3_misinformation.router import router as c3_router
from app.core.database import get_session


CLAIM_ID = UUID("40404040-4040-4040-8040-404040404040")
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


class FakeTransaction:
    def __init__(self):
        self.entered = False
        self.rolled_back = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.rolled_back = exc_type is not None
        return False


class FakeSession:
    def __init__(self):
        self.transaction = FakeTransaction()
        self.closed = False

    def begin(self):
        return self.transaction


def completed_claim(*, status="completed", result=None):
    return {
        "claim_id": CLAIM_ID,
        "processing_status": status,
        "queued_at": NOW,
        "started_at": NOW if status in {"processing", "completed", "failed"} else None,
        "completed_at": NOW if status == "completed" else None,
        "failed_at": NOW if status == "failed" else None,
        "result_payload": result,
    }


class AsyncC3HttpIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(c3_router, prefix="/api/v1/c3")
        self.sessions = []

        async def session_dependency():
            session = FakeSession()
            self.sessions.append(session)
            try:
                yield session
            finally:
                session.closed = True

        self.app.dependency_overrides[get_session] = session_dependency
        self.set_token_validator(lambda texts: [len(text) for text in texts])

    def set_token_validator(self, validator):
        async def dependency():
            return validator
        self.app.dependency_overrides[api.get_claim_token_validator] = dependency

    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.app.dependency_overrides.clear()

    async def test_submission_queues_claim_preserves_source_and_avoids_worker(self):
        create_claim = AsyncMock(return_value=CLAIM_ID)
        with patch.object(api.repository, "create_text_claim", create_claim), \
             patch.object(api, "_detect_claim_language", return_value=None):
            response = await self.client.post(
                "/api/v1/c3/claims/analyze", json={"text": "  Sinhala claim text.\t"},
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {"claim_id": str(CLAIM_ID), "status": "processing"})
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIn(COOKIE_NAME, response.cookies)
        create_claim.assert_awaited_once()
        self.assertEqual(create_claim.await_args.args[0], "  Sinhala claim text.\t")
        self.assertEqual(create_claim.await_args.kwargs["cleaned_text"], "Sinhala claim text.")
        self.assertTrue(self.sessions[0].transaction.entered)
        self.assertFalse(self.sessions[0].transaction.rolled_back)
        self.assertTrue(self.sessions[0].closed)

    async def test_submission_validation_rejects_blank_invalid_and_extra_fields(self):
        with patch.object(api.repository, "create_text_claim", new=AsyncMock()) as create_claim:
            for body in ({}, {"text": None}, {"text": 3}, {"text": "claim", "extra": 1}, []):
                response = await self.client.post("/api/v1/c3/claims/analyze", json=body)
                self.assertEqual(response.status_code, 422)
            for text in ("", " \t\n ", "\ufeff"):
                response = await self.client.post(
                    "/api/v1/c3/claims/analyze", json={"text": text},
                )
                self.assertEqual(response.status_code, 400)
        create_claim.assert_not_awaited()

    async def test_submission_tokenizer_and_storage_failures_are_sanitized(self):
        def unavailable(_texts):
            raise api.TokenizerUnavailableError("private tokenizer detail")

        self.set_token_validator(unavailable)
        response = await self.client.post(
            "/api/v1/c3/claims/analyze", json={"text": "claim"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private tokenizer detail", response.text)

        self.set_token_validator(
            lambda _texts: (_ for _ in ()).throw(
                api.EmbeddingInputTooLongError(index=0, token_count=99, limit=10),
            )
        )
        response = await self.client.post(
            "/api/v1/c3/claims/analyze", json={"text": "claim"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "claim_text_too_long")

        self.set_token_validator(lambda _texts: [1])
        with patch.object(
            api.repository, "create_text_claim",
            new=AsyncMock(side_effect=RuntimeError("private persistence detail")),
        ):
            response = await self.client.post(
                "/api/v1/c3/claims/analyze", json={"text": "claim"},
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["code"], "claim_persistence_failed")
        self.assertNotIn("private persistence detail", response.text)
        self.assertNotIn("set-cookie", response.headers)
        session = self.sessions[-1]
        self.assertTrue(session.transaction.rolled_back)

    async def test_existing_and_unknown_cookie_submission_contract(self):
        existing = "A" * 43
        self.client.cookies.set(COOKIE_NAME, existing)
        with patch.object(api.repository, "anonymous_session_has_claim", new=AsyncMock(return_value=True)), \
             patch.object(api.repository, "create_text_claim", new=AsyncMock(return_value=CLAIM_ID)), \
             patch.object(api, "_detect_claim_language", return_value=None):
            response = await self.client.post(
                "/api/v1/c3/claims/analyze", json={"text": "claim"},
            )
        self.assertEqual(response.status_code, 202)
        self.assertNotIn("set-cookie", response.headers)

        self.client.cookies.clear()
        unknown = "B" * 43
        with patch.object(api.repository, "anonymous_session_has_claim", new=AsyncMock(return_value=False)), \
             patch.object(api.repository, "create_text_claim", new=AsyncMock(return_value=CLAIM_ID)), \
             patch.object(api, "_detect_claim_language", return_value=None):
            rotated = await self.client.post(
                "/api/v1/c3/claims/analyze",
                json={"text": "claim"}, headers={"Cookie": f"{COOKIE_NAME}={unknown}"},
            )
        self.assertEqual(rotated.status_code, 202)
        self.assertIn(COOKIE_NAME, rotated.cookies)
        self.assertNotEqual(rotated.cookies[COOKIE_NAME], unknown)

    async def test_result_requires_owner_and_preserves_completed_snapshot(self):
        missing = await self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.json()["detail"]["code"], "invalid_claim_session")
        self.assertEqual(missing.headers["cache-control"], "no-store")

        self.client.cookies.set(COOKIE_NAME, "A" * 43)
        stored = {
            "mode": "claim",
            "claim_verdict": "Misleading",
            "citations": ["chunk:LEGAL-001"],
            "insufficient_evidence": False,
            "explanation": "Fixture evidence narrows the claim.",
            "evidence": {"passages": [{"evidence_id": "chunk:LEGAL-001", "metadata": {"title": "Act"}}]},
        }
        with patch.object(
            api.repository, "get_owned_claim_result",
            new=AsyncMock(return_value=completed_claim(result=stored)),
        ) as lookup:
            response = await self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(response.json()["result"], stored)
        self.assertEqual(response.headers["cache-control"], "no-store")
        lookup.assert_awaited_once()

    async def test_result_lifecycle_states_and_foreign_or_malformed_data(self):
        self.client.cookies.set(COOKIE_NAME, "A" * 43)
        for status in ("queued", "processing", "failed"):
            with self.subTest(status=status), patch.object(
                api.repository, "get_owned_claim_result",
                new=AsyncMock(return_value=completed_claim(status=status)),
            ):
                response = await self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], status)
            self.assertIsNone(response.json()["result"])

        with patch.object(
            api.repository, "get_owned_claim_result", new=AsyncMock(return_value=None),
        ):
            foreign = await self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
        self.assertEqual(foreign.status_code, 404)
        self.assertEqual(foreign.json()["detail"]["code"], "result_not_found")

        with patch.object(
            api.repository, "get_owned_claim_result",
            new=AsyncMock(return_value=completed_claim(status="bogus")),
        ):
            malformed = await self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
        self.assertEqual(malformed.status_code, 500)
        self.assertEqual(malformed.json()["detail"]["code"], "result_state_unavailable")

    async def test_clients_are_isolated_and_multilingual_source_contract_survives(self):
        client_b = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://testserver",
        )
        try:
            with patch.object(api.repository, "create_text_claim", new=AsyncMock(return_value=CLAIM_ID)), \
                 patch.object(api, "_detect_claim_language", return_value="si"):
                submitted = await self.client.post(
                    "/api/v1/c3/claims/analyze", json={"text": "සිංහල ප්‍රකාශය."},
                )
            self.assertEqual(submitted.status_code, 202)
            self.assertIn(COOKIE_NAME, submitted.cookies)
            foreign = await client_b.get(f"/api/v1/c3/results/{CLAIM_ID}")
            self.assertEqual(foreign.status_code, 401)

            with patch.object(api.repository, "get_owned_claim_result", new=AsyncMock(return_value=completed_claim(result={"claim": "සිංහල ප්‍රකාශය."}))):
                owned = await self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
            self.assertEqual(owned.status_code, 200)
            self.assertEqual(owned.json()["result"]["claim"], "සිංහල ප්‍රකාශය.")
        finally:
            await client_b.aclose()


if __name__ == "__main__":
    unittest.main()
