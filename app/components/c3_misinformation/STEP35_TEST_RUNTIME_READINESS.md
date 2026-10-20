# Step 35 — C3 test runtime readiness

## Runtime diagnosis

The group environment is `/home/sano_01/J26-IT-338-Backend/.venv` with
Python 3.12.3. The following bounded diagnostics were run without network,
database, provider, or model access:

- basic Python thread: completed and exited 0;
- `asyncio.run`: completed and exited 0;
- `asyncio.to_thread`: function returned, but process shutdown timed out;
- minimal FastAPI `TestClient`: timed out before the handler/request completed;
- AnyIO blocking portal: portal opened, then portal call timed out;
- HTTPX `AsyncClient` + `ASGITransport`: returned HTTP 200 and exited 0.

The minimal TestClient and portal failures are application-independent. The
current execution environment is not equivalent to the user's normal Ubuntu
terminal; Codex cannot execute commands in that separate terminal. To compare
there, run the same short diagnostics from the repository directory using
`timeout 10s ./.venv/bin/python ...`. No claim is made that the two execution
contexts are identical.

## Confirmed C3 defect and correction

The previous C3 synchronous-boundary bridges used
`loop.call_soon_threadsafe(...)` to complete an asyncio Future from daemon
threads. In this runtime, the provider thread completed and the callback was
queued, but the event loop remained asleep in its selector and the awaiting
task never resumed. Faulthandler showed no active provider work while the loop
waited; a standalone callback diagnostic worked, isolating the failure to the
bridge behavior rather than provider code.

The smallest C3-only correction changes the three affected bridges in
[rag_service_async.py](./rag_service_async.py),
[multilingual_claim_adapter.py](./pipeline/multilingual_claim_adapter.py),
and [worker_dependencies.py](./worker_dependencies.py) to poll a completion
event with short cancellable asyncio sleeps. Results and exceptions are still
captured from the synchronous thread, completed calls are joined briefly, and
cancellation still propagates without pretending to stop an already-running
provider. No API, claim, evidence, worker lease, ownership, or persistence
contract changed.

## Threading and cancellation safety

- One daemon thread is created per explicitly injected synchronous boundary.
- No default executor is created or shut down by these bridges.
- Cancellation leaves only the non-interruptible provider/tokenizer call in
  flight; it cannot write to the database because the bridge has no persistence
  capability.
- The awaiting coroutine is cancelled immediately; no late result is returned
  to RAG or the worker.
- Exceptions are transferred through shared completion state and re-raised.
- No event-loop callback is attempted after loop closure.
- Worker result writes remain separately lease/ownership-fenced.
- Provider/tokenizer-specific timeouts and deployment concurrency limits remain
  required; daemon threads are not a substitute for those limits.

## Focused verification

Passed in the group `.venv`:

- RAG service: **14 tests passed**
- Claim worker + runner: **23 tests passed**
- Worker dependencies: **11 tests passed**
- Multilingual adapter: **7 tests passed**
- Worker RAG adapter: **3 tests passed**
- Repository/model/lifecycle: **33 tests passed**
- Async HTTP integration: **7 tests passed**
- Step 25 evaluation: **15 tests passed**
- Shared corpus: **10 tests passed**
- `pip check`, C3 imports, syntax compilation, and `git diff --check`

## Remaining regression blocker

The complete C3 discovery suite was attempted once with a 60-second external
timeout and exited **124**. A verbose bounded run identified the first blocked
method as:

`ClaimApiTests.test_dynamic_model_token_limit_and_unavailable_tokenizer_errors`

The independent minimal TestClient reproduction hangs before request handling,
so this is not evidence of a C3 API defect. The synchronous TestClient/AnyIO
environment limitation remains; the ASGITransport-based HTTP suite is the
working offline HTTP verification path. No timed-out command is counted as
passed, and no tests were skipped or weakened.

## Alembic and environment checks

- `pip check`: passed;
- Alembic heads: single head `c3_0001`;
- Alembic history: `shared_0001 → c3_0001`;
- no migration, database, Neon, provider, or model operation was executed.

## Neon and next-step readiness

Offline C3 regression evidence is substantially restored, but full discovery
evidence is not available until TestClient/AnyIO is rerun in a corrected
runtime or replaced by an approved equivalent harness. Isolated PostgreSQL
verification can be planned next only with an authorized disposable database,
applied schema ownership, and a test runtime that records the remaining
TestClient limitation rather than hiding it.
