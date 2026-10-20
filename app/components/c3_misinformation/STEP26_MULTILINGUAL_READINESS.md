# Step 26 — Multilingual Claim Processing Readiness

## Implemented architecture

Claim processing keeps the submitted claim authoritative. `MultilingualClaimAdapter`
optionally detects the source language and creates a separate English
`retrieval_query`; it never replaces `ClaimInput.claim`. Detection and translation
are dependency-injected and have no import-time model or network work.

## Language flows

- English (`en`) claims use the original text for retrieval and assessment.
- Sinhala (`si`) and Tamil (`ta`) claims may use an explicitly injected
  Sinhala/Tamil-to-English provider for retrieval only.
- Mixed, uncertain, unsupported, and unrecognized languages remain untranslated
  and use the original text as the retrieval query.
- Empty claims remain invalid at the RAG boundary.

`COMPNET_LANGUAGE_DETECTION_ENABLED` controls detection. It defaults to disabled;
accepted enabled values are `1`, `true`, `yes`, and `on` (case-insensitive), and
accepted disabled values are `0`, `false`, `no`, `off`, or empty. Invalid values
fail configuration validation. Translation is never enabled by this flag alone:
an adapter and provider must be explicitly composed.

## RAG and worker integration

`AsyncRagService` validates both the source claim and any translated retrieval
query, validates token limits for both, retrieves using only the retrieval query,
and builds the RAG assessment prompt from the original claim. Synchronous
translation/detection providers are offloaded from the event loop through the
adapter's async boundary. Question mode is unchanged. The worker adapter keeps
the existing analyzer signature and passes through lease, ownership, and
cancellation behavior unchanged.

Evidence packets, metadata, ranking, and citation IDs are passed through without
translation or rewriting. Empty evidence produces a local, schema-validated
abstention without a hosted model call.

## Errors and limitations

Invalid, empty, or malformed provider translations are rejected. Provider
unavailability and unexpected provider failures fall back to the original claim
and are never silently presented as a translation. Cancellation propagates from
retrieval and hosted analysis. The Gemini translation adapter is experimental,
requires an explicitly configured client/API key at runtime, and does not prove
translation or legal accuracy.

The current legal corpus and retrieval index are English-oriented. Sinhala/Tamil
retrieval therefore depends on the quality of the optional translation and may
miss language-specific terminology, local drafting conventions, or sources not
represented in the English corpus.

## Offline verification

Step 26 tests use mocked identifiers, translation providers, evidence builders,
token validators, and model providers. They cover English, Sinhala, and Tamil
flows; source-text preservation; retrieval-query translation; feature flags;
unknown/unsupported languages; invalid and failed translations; token checks;
worker adaptation; evidence/citation preservation; cancellation; provider
unavailability; empty evidence; and import-time/no-eager-model behavior.

The focused multilingual module completed **7 tests, OK** with the repository
interpreter. The focused multilingual/RAG/worker/dependency group completed
**24 tests, OK**. The separate Step 25 evaluation suite completed **15 tests,
OK**, and shared-corpus tests completed **10 tests, OK**.

The original async hang was traced to synchronous `asyncio.to_thread` executor
shutdown under this Python 3.12 environment, including an event-wait fixture.
The async bridges now use bounded daemon threads, propagate exceptions, join
completed calls, preserve cancellation for in-flight calls, and avoid callbacks
after loop closure. The cancellation fixture uses non-executor event polling.

The exact full C3 discovery command remains blocked by this environment's
`fastapi.testclient.TestClient`: even a minimal FastAPI/TestClient request hangs
before the handler runs, independently of C3 code. It is not reported as a
passing regression result.

## TestClient blocker investigation

The failure is independently reproducible outside the repository with FastAPI
0.141.1, Starlette 1.6.0, AnyIO 4.15.1, HTTPX 0.28.1, and Python 3.12.3:

- A minimal FastAPI app reaches `TestClient` construction but hangs at the first
  synchronous request; its handler is never entered.
- A minimal `httpx.AsyncClient` with `ASGITransport` completes successfully with
  HTTP 200.
- A minimal synchronous thread and basic `asyncio.run(asyncio.sleep(0))`
  complete successfully.
- `asyncio.to_thread()` returns its value but the process hangs during executor
  shutdown.
- `anyio.from_thread.start_blocking_portal()` hangs waiting for its portal
  future, before application code runs. `faulthandler` shows the portal thread
  blocked in the AnyIO asyncio runner's selector and the caller waiting on the
  portal future.

This confirms an application-independent AnyIO/TestClient runtime problem. The
installed versions are internally version-compatible by declared package
requirements, but this environment's portal/executor behavior is not usable for
the synchronous TestClient regression harness. No package changes were made.

## Threading safety assessment

The C3 bridges use daemon threads only for explicitly synchronous, injected
boundaries. Successful and failed calls wait for thread completion and join the
thread. Cancellation does not pretend to stop an already-running provider;
that call may continue until the provider's own timeout. Closed-loop callbacks
are suppressed, exceptions are delivered to the awaiting task, and no detached
thread performs database access or persistence. The worker's ownership-fenced
write path remains unchanged.

This remains a runtime compromise: a provider that ignores its own timeout can
consume one daemon thread after cancellation. Production deployment must provide
bounded provider/tokenizer timeouts and concurrency limits; offline tests do not
establish translation accuracy or hosted-provider behavior.

The full C3 discovery command is therefore blocked only by the independent
TestClient/AnyIO environment issue. Step 27 may proceed only with this testing
limitation explicitly accepted and with synchronous TestClient coverage rerun in
a corrected runtime; it must not be treated as hidden full-regression evidence.

The focused and regression commands use the repository interpreter:

```text
PYTHONPATH=. /home/sano_01/fyp/backend/.venv/bin/python -m unittest ...
```

Runtime requirements remain an applied database schema, locally cached approved
LaBSE weights, configured Gemini dependencies/credentials when hosted analysis
is desired, and a correctly configured language identifier/provider. None of
these runtime resources are loaded or contacted by offline imports or tests.

Translation quality, legal fidelity, and completeness of multilingual coverage
remain unverified until representative Sinhala/Tamil fixtures and runtime
providers are evaluated by the project team.
