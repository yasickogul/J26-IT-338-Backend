# Step 27 — Async HTTP Integration Readiness

## Test architecture

The new `test_async_http_integration.py` suite uses `unittest.IsolatedAsyncioTestCase`,
`httpx.AsyncClient`, and `httpx.ASGITransport` against the real C3 FastAPI
router. It does not use Starlette/FastAPI `TestClient`, PostgreSQL, Gemini, a
worker process, or model weights. Session, repository, token-validation, and
language-detection boundaries are dependency-overridden or mocked.

## Verified HTTP behavior

`POST /api/v1/c3/claims/analyze` is covered for valid 202 acknowledgements,
claim ID/status serialization, queued transaction setup, original raw text and
cleaned text preservation, blank/invalid/extra-field rejection, token-limit and
tokenizer failures, persistence failure sanitization, rollback behavior,
no-store headers, cookie issuance, cookie reuse, unknown-token rotation, and
absence of worker/RAG/Gemini execution during submission.

`GET /api/v1/c3/results/{claim_id}` is covered for missing/owned/foreign
sessions, queued/processing/completed/failed states, malformed lifecycle state,
completed result serialization, evidence metadata and citation preservation,
sanitized lookup errors, and no-store headers. Missing and foreign claims expose
no ownership information.

Separate async clients verify anonymous-session isolation: client B cannot read
client A's claim, while client A can retrieve its own stored snapshot. Cookie
values and hashes are not logged or included in diagnostics.

## Offline lifecycle and multilingual compatibility

The tests exercise the real HTTP routes through FastAPI's ASGI application and
mock only persistence/provider boundaries. They verify the submission-to-queued
contract and retrieval serialization. A full database-backed worker vertical
slice remains a runtime concern and is not fabricated in the test suite.

Sinhala source text remains unchanged in the HTTP contract; lifecycle status,
ownership, result shape, evidence IDs, and citation IDs remain compatible with
Step 26. Translation and extraction remain disabled unless explicitly injected.

## Environment limitation

The existing synchronous `TestClient` regression suite remains blocked by the
independently reproducible AnyIO/TestClient portal problem documented in
`STEP26_MULTILINGUAL_READINESS.md`. The new ASGITransport tests avoid that
portal and complete successfully; this does not mean the old TestClient suite
passed.

## Verification

Focused Step 27 tests completed **7 tests, OK**. Step 26 focused tests remain
green separately. Step 25 evaluation completed **15 tests, OK** and shared
corpus tests completed **10 tests, OK**. Full C3 discovery remains blocked at
the pre-existing TestClient-based API tests.

The API's local tokenizer and language detection are synchronous in this
environment because Starlette/AnyIO executor bridges are not usable under the
ASGI test runtime. Production deployment should provide bounded tokenizer and
provider work, explicit concurrency limits, and a functioning AnyIO/TestClient
runtime before relying on synchronous HTTP regression coverage.

No translation or legal accuracy claim is made from mocked offline tests.
