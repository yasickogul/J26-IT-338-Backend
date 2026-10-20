"""Offline HTTP contract checks for C3 claim lifecycle routes."""

from datetime import datetime, timezone
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.components.c3_misinformation import api
from app.components.c3_misinformation.anonymous_claim_session import (
    COOKIE_NAME,
    SECURE_COOKIE_NAME,
    session_token_hash,
)
from app.components.c3_misinformation.claim_lifecycle_contract import ClaimStorageError
from app.components.c3_misinformation.pipeline.claim_input_tokens import TokenizerUnavailableError
from app.core.database import get_session


CLAIM_ID = UUID("03030303-0303-4303-8303-030303030303")
OWNER_TOKEN = "A" * 43
OWNER_HASH = session_token_hash(OWNER_TOKEN)


class FakeTransaction:
    def __init__(self, *, fail_commit=False):
        self.fail_commit = fail_commit
        self.entered = False
        self.exited_with = None
        self.rolled_back = False

    async def __aenter__(self):
        self.entered = True

    async def __aexit__(self, exc_type, exc, traceback):
        self.exited_with = exc_type
        if exc_type is None and self.fail_commit:
            self.rolled_back = True
            raise RuntimeError("private commit detail")
        self.rolled_back = exc_type is not None
        return False


class FakeSession:
    def __init__(self, *, fail_commit=False):
        self.transaction = FakeTransaction(fail_commit=fail_commit)
        self.closed = False

    def begin(self):
        return self.transaction


class ClaimApiTests(unittest.TestCase):
    def setUp(self):
        self.sessions = []
        self.app = FastAPI()
        self.app.include_router(api.router, prefix="/api/v1/c3")

        async def session_dependency():
            session = FakeSession()
            self.sessions.append(session)
            try:
                yield session
            finally:
                session.closed = True

        self.app.dependency_overrides[get_session] = session_dependency
        self.app.dependency_overrides[api.get_claim_token_validator] = lambda: (
            lambda texts: [len(text) for text in texts]
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.dependency_overrides.clear()

    def test_submission_creates_claim_commits_and_issues_secure_opaque_cookie(self):
        create_claim = AsyncMock(return_value=CLAIM_ID)
        session_has_claim = AsyncMock(return_value=False)
        with patch.object(api.repository, "create_text_claim", create_claim), \
             patch.object(api.repository, "anonymous_session_has_claim", session_has_claim), \
             patch.object(api, "_detect_claim_language", return_value=None), \
             patch("app.components.c3_misinformation.claim_worker.process_one_claim") as worker, \
             patch("app.components.c3_misinformation.rag_service_async.AsyncRagService.run") as rag, \
             patch("app.components.c3_misinformation.rag_service_async.ThreadedGeminiProvider.generate_json") as gemini:
            response = self.client.post(
                "/api/v1/c3/claims/analyze",
                json={"text": "  A legal claim\t"},
                headers={"Cookie": f"{COOKIE_NAME}=invalid"},
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {"claim_id": str(CLAIM_ID), "status": "processing"})
        create_claim.assert_awaited_once()
        self.assertEqual(create_claim.await_args.args, ("  A legal claim\t",))
        self.assertEqual(create_claim.await_args.kwargs["cleaned_text"], "A legal claim")
        self.assertEqual(
            create_claim.await_args.kwargs["anonymous_session_hash"],
            session_token_hash(response.cookies[COOKIE_NAME]),
        )
        self.assertTrue(self.sessions[0].transaction.entered)
        self.assertIsNone(self.sessions[0].transaction.exited_with)
        self.assertTrue(self.sessions[0].closed)
        cookie = response.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        self.assertIn("path=/", cookie)
        self.assertNotIn("secure", cookie)
        self.assertNotIn("max-age", cookie)
        self.assertNotIn("expires", cookie)
        worker.assert_not_called()
        rag.assert_not_called()
        gemini.assert_not_called()

    def test_existing_owned_cookie_is_reused_without_reissue(self):
        self.client.cookies.set(COOKIE_NAME, OWNER_TOKEN)
        with patch.object(api.repository, "anonymous_session_has_claim", new=AsyncMock(return_value=True)), \
             patch.object(api.repository, "create_text_claim", new=AsyncMock(return_value=CLAIM_ID)) as create_claim, \
             patch.object(api, "_detect_claim_language", return_value=None):
            response = self.client.post(
                "/api/v1/c3/claims/analyze",
                json={"text": "valid claim"},
            )
        self.assertEqual(response.status_code, 202)
        self.assertNotIn("set-cookie", response.headers)
        self.assertEqual(create_claim.await_args.kwargs["anonymous_session_hash"], OWNER_HASH)

        self.client.cookies.clear()
        unknown_token = "B" * 43
        with patch.object(api.repository, "anonymous_session_has_claim", new=AsyncMock(return_value=False)), \
             patch.object(api.repository, "create_text_claim", new=AsyncMock(return_value=CLAIM_ID)) as create_claim, \
             patch.object(api, "_detect_claim_language", return_value=None):
            rotated = self.client.post(
                "/api/v1/c3/claims/analyze",
                json={"text": "valid claim"},
                headers={"Cookie": f"{COOKIE_NAME}={unknown_token}"},
            )
        self.assertEqual(rotated.status_code, 202)
        self.assertNotEqual(rotated.cookies[COOKIE_NAME], unknown_token)
        self.assertEqual(
            create_claim.await_args.kwargs["anonymous_session_hash"],
            session_token_hash(rotated.cookies[COOKIE_NAME]),
        )

    def test_https_submission_uses_host_only_secure_cookie(self):
        https_client = TestClient(self.app, base_url="https://testserver")
        try:
            with patch.object(
                api.repository, "create_text_claim", new=AsyncMock(return_value=CLAIM_ID),
            ), patch.object(api, "_detect_claim_language", return_value=None):
                response = https_client.post(
                    "/api/v1/c3/claims/analyze",
                    json={"text": "valid claim"},
                )
        finally:
            https_client.close()
        self.assertEqual(response.status_code, 202)
        cookie = response.headers["set-cookie"].lower()
        self.assertTrue(cookie.startswith(f"{SECURE_COOKIE_NAME}=".lower()))
        self.assertIn("secure", cookie)
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        self.assertIn("path=/", cookie)
        self.assertNotIn("domain=", cookie)

    def test_invalid_body_and_blank_claim_are_rejected_before_storage(self):
        with patch.object(api.repository, "create_text_claim", new=AsyncMock()) as create_claim:
            for body in ({}, {"text": None}, {"text": 3}, {"text": "claim", "extra": 1}, []):
                with self.subTest(body=body):
                    response = self.client.post("/api/v1/c3/claims/analyze", json=body)
                    self.assertEqual(response.status_code, 422)
            for text in ("", " \t\n ", "\ufeff"):
                with self.subTest(text=repr(text)):
                    response = self.client.post(
                        "/api/v1/c3/claims/analyze",
                        json={"text": text},
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json()["detail"], {
                        "code": "invalid_claim_text", "category": "input",
                    })
        create_claim.assert_not_awaited()

    def test_dynamic_model_token_limit_and_unavailable_tokenizer_errors(self):
        limit = 5

        def exact_boundary_validator(texts):
            if any(len(text) > limit for text in texts):
                raise api.EmbeddingInputTooLongError(
                    index=0, token_count=limit + 1, limit=limit,
                )
            return [len(text) for text in texts]

        self.app.dependency_overrides[api.get_claim_token_validator] = lambda: exact_boundary_validator
        with patch.object(api.repository, "create_text_claim", new=AsyncMock(return_value=CLAIM_ID)), \
             patch.object(api, "_detect_claim_language", return_value=None):
            accepted = self.client.post(
                "/api/v1/c3/claims/analyze",
                json={"text": "claim"},
            )
        self.assertEqual(accepted.status_code, 202)

        self.app.dependency_overrides[api.get_claim_token_validator] = lambda: (
            lambda _texts: (_ for _ in ()).throw(
                api.EmbeddingInputTooLongError(index=0, token_count=9, limit=8)
            )
        )
        with patch.object(api.repository, "create_text_claim", new=AsyncMock()) as create_claim:
            response = self.client.post(
                "/api/v1/c3/claims/analyze",
                json={"text": "long legal claim"},
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], {
            "code": "claim_text_too_long", "category": "input", "limit": 8,
        })
        create_claim.assert_not_awaited()

        self.app.dependency_overrides[api.get_claim_token_validator] = lambda: (
            lambda _texts: (_ for _ in ()).throw(TokenizerUnavailableError("private path"))
        )
        unavailable = self.client.post(
            "/api/v1/c3/claims/analyze",
            json={"text": "legal claim"},
        )
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["detail"], {
            "code": "claim_tokenizer_unavailable", "category": "preparation",
        })
        self.assertNotIn("private path", unavailable.text)

    def test_storage_failure_rolls_back_and_does_not_acknowledge(self):
        session = FakeSession(fail_commit=True)
        self.sessions.append(session)

        async def fail_commit_session():
            try:
                yield session
            finally:
                session.closed = True

        self.app.dependency_overrides[get_session] = fail_commit_session
        with patch.object(api.repository, "create_text_claim", new=AsyncMock(return_value=CLAIM_ID)), \
             patch.object(api, "_detect_claim_language", return_value=None):
            response = self.client.post(
                "/api/v1/c3/claims/analyze",
                json={"text": "legal claim"},
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], {
            "code": "claim_persistence_failed", "category": "storage",
        })
        self.assertNotIn("claim_id", response.json())
        self.assertNotIn("private commit detail", response.text)
        self.assertIsNone(session.transaction.exited_with)
        self.assertTrue(session.transaction.rolled_back)
        self.assertTrue(session.closed)
        self.assertNotIn("set-cookie", response.headers)

    def test_status_and_completed_result_preserve_snapshot_and_hide_failure_metadata(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        snapshot = {
            "mode": "claim", "claim_verdict": None, "extra": {"preserved": [1, 2]},
            "evidence": {"passages": [{"chunk_id": "legal-chunk-1"}]},
        }
        self.client.cookies.set(COOKIE_NAME, OWNER_TOKEN)
        for state in ("queued", "processing", "failed", "completed"):
            claim = {
                "claim_id": CLAIM_ID,
                "processing_status": state,
                "queued_at": now,
                "started_at": None,
                "completed_at": now if state == "completed" else None,
                "failed_at": now if state == "failed" else None,
                "failure_category": "provider" if state == "failed" else None,
                "result_payload": snapshot if state == "completed" else None,
            }
            with self.subTest(state=state), patch.object(
                api.repository, "get_owned_claim_result", new=AsyncMock(return_value=claim),
            ):
                response = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], state)
            self.assertEqual(
                response.json()["result"],
                snapshot if state == "completed" else None,
            )
            self.assertNotIn("failure_category", response.json())
            self.assertEqual(response.headers["cache-control"], "no-store")

    def test_missing_foreign_and_invalid_sessions_are_protected(self):
        self.client.cookies.set(COOKIE_NAME, OWNER_TOKEN)
        with patch.object(api.repository, "get_owned_claim_result", new=AsyncMock(return_value=None)) as reader:
            missing = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
            foreign = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
        self.assertEqual(missing.status_code, foreign.status_code)
        self.assertEqual(missing.json(), foreign.json())
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["detail"]["code"], "result_not_found")
        reader.assert_awaited()
        self.assertEqual(reader.await_args.args, (CLAIM_ID, OWNER_HASH))

        self.client.cookies.clear()
        with patch.object(api.repository, "get_owned_claim_result", new=AsyncMock()) as reader:
            unauthenticated = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
        self.assertEqual(unauthenticated.status_code, 401)
        reader.assert_not_awaited()

    def test_lookup_failure_is_sanitized_and_https_cookie_name_is_used(self):
        https_client = TestClient(self.app, base_url="https://testserver")
        https_client.cookies.set(SECURE_COOKIE_NAME, OWNER_TOKEN)
        try:
            with patch.object(
                api.repository, "get_owned_claim_result",
                new=AsyncMock(side_effect=ClaimStorageError("private database detail")),
            ):
                response = https_client.get(f"/api/v1/c3/results/{CLAIM_ID}")
        finally:
            https_client.close()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["code"], "result_lookup_failed")
        self.assertNotIn("private database detail", response.text)


if __name__ == "__main__":
    unittest.main()
