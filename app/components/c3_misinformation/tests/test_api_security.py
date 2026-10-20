"""Focused offline security and malformed-storage checks for C3 routes."""

from unittest import IsolatedAsyncioTestCase
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
from app.core.database import get_session


CLAIM_ID = UUID("33333333-3333-4333-8333-333333333333")
KNOWN_TOKEN = "A" * 43
UNKNOWN_TOKEN = "B" * 43


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSession:
    def begin(self):
        return FakeTransaction()


class ApiSecurityTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.sessions = []
        self.app = FastAPI()
        self.app.include_router(api.router, prefix="/api/v1/c3")

        async def session_dependency():
            session = FakeSession()
            self.sessions.append(session)
            yield session

        self.app.dependency_overrides[get_session] = session_dependency
        self.app.dependency_overrides[api.get_claim_token_validator] = lambda: (
            lambda texts: [len(text) for text in texts]
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.dependency_overrides.clear()

    def test_missing_malformed_and_unknown_sessions_cannot_read_claims(self):
        with patch.object(
            api.repository,
            "get_owned_claim_result",
            new=AsyncMock(return_value=None),
        ) as lookup:
            missing = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
            self.client.cookies.set(COOKIE_NAME, "malformed")
            malformed = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")
            lookup.assert_not_awaited()
            self.client.cookies.set(COOKIE_NAME, UNKNOWN_TOKEN)
            unknown = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(malformed.status_code, 401)
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json(), {"detail": {"code": "result_not_found"}})
        self.assertEqual(unknown.headers["cache-control"], "no-store")
        lookup.assert_awaited_once_with(
            CLAIM_ID, session_token_hash(UNKNOWN_TOKEN), session=self.sessions[-1],
        )

    def test_unowned_well_formed_token_is_rotated_before_new_claim_creation(self):
        issued = "C" * 43
        self.app.dependency_overrides[api.get_claim_token_validator] = lambda: (
            lambda texts: [len(text) for text in texts]
        )
        self.client.cookies.set(COOKIE_NAME, UNKNOWN_TOKEN)
        with patch.object(api, "new_session_token", return_value=issued), \
             patch.object(
                 api.repository,
                 "anonymous_session_has_claim",
                 new=AsyncMock(return_value=False),
             ), patch.object(
                 api.repository,
                 "create_text_claim",
                 new=AsyncMock(return_value=CLAIM_ID),
             ) as create_claim, patch.object(api, "_detect_claim_language", return_value=None):
            response = self.client.post(
                "/api/v1/c3/claims/analyze",
                json={"text": "A valid legal claim."},
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.cookies[COOKIE_NAME], issued)
        self.assertEqual(
            create_claim.await_args.kwargs["anonymous_session_hash"],
            session_token_hash(issued),
        )
        self.assertNotEqual(
            create_claim.await_args.kwargs["anonymous_session_hash"],
            session_token_hash(UNKNOWN_TOKEN),
        )
        self.assertIn("httponly", response.headers["set-cookie"].lower())
        self.assertIn("samesite=lax", response.headers["set-cookie"].lower())
        self.assertNotIn("secure", response.headers["set-cookie"].lower())

    def test_secure_cookie_configuration_applies_on_http(self):
        with patch.dict("os.environ", {"COMPNET_SECURE_COOKIES": "1"}), \
             patch.object(
                 api.repository,
                 "create_text_claim",
                 new=AsyncMock(return_value=CLAIM_ID),
             ), patch.object(api, "_detect_claim_language", return_value=None):
            response = self.client.post(
                "/api/v1/c3/claims/analyze",
                json={"text": "A valid legal claim."},
            )

        cookie = response.headers["set-cookie"].lower()
        self.assertTrue(cookie.startswith(SECURE_COOKIE_NAME.lower() + "="))
        self.assertIn("secure", cookie)
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        self.assertIn("path=/", cookie)
        self.assertNotIn("domain=", cookie)

    def test_malformed_and_non_json_requests_do_not_reach_storage(self):
        with patch.object(api.repository, "create_text_claim", new=AsyncMock()) as create_claim:
            malformed_json = self.client.post(
                "/api/v1/c3/claims/analyze",
                content=b'{"text":',
                headers={"Content-Type": "application/json"},
            )
            form_body = self.client.post(
                "/api/v1/c3/claims/analyze",
                content=b"text=legal+claim",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        self.assertEqual(malformed_json.status_code, 422)
        self.assertEqual(form_body.status_code, 422)
        create_claim.assert_not_awaited()

    def test_malformed_owned_result_is_sanitized_and_not_cached(self):
        self.client.cookies.set(COOKIE_NAME, KNOWN_TOKEN)
        with patch.object(
            api.repository,
            "get_owned_claim_result",
            new=AsyncMock(return_value={
                "claim_id": CLAIM_ID,
                "queued_at": None,
                "started_at": None,
                "completed_at": None,
                "failed_at": None,
                "result_payload": None,
            }),
        ):
            response = self.client.get(f"/api/v1/c3/results/{CLAIM_ID}")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": {"code": "result_state_unavailable"}},
        )
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("Traceback", response.text)


if __name__ == "__main__":
    import unittest

    unittest.main()
