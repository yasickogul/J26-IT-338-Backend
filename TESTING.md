# Backend testing

## Install

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
uv pip install -r requirements-dev.txt
```

Python **3.11+**. Dependencies are managed via `requirements.txt` / `requirements-dev.txt` (no lockfile in this repo yet).

## Default offline suite (CI)

```bash
pytest
# equivalent:
pytest -m "not integration and not live_llm"
```

These runs:

- Do **not** connect to Neon / production Postgres
- Do **not** download embedding models or regenerate vectors
- Do **not** call LM Studio or paid LLM APIs
- Clear `DATABASE_URL` / `DATABASE_URL_DIRECT` unless `RUN_INTEGRATION_TESTS=1`

### Markers

| Marker | Purpose |
|--------|---------|
| `unit` | Offline tests (default) |
| `integration` | Needs a real Postgres (+ pgvector when required). Excluded from default CI. |
| `live_llm` | Contacts LM Studio / cloud LLM. Excluded from default CI. |

## Integration tests (opt-in)

Use an **isolated** Postgres with pgvector — never the production Neon URL.

```bash
export RUN_INTEGRATION_TESTS=1
export DATABASE_URL='postgresql+asyncpg://…'          # test DB only
export DATABASE_URL_DIRECT='postgresql+asyncpg://…'   # test DB only
pytest -m integration
```

Stage 2 persistence / PDF tests under `app/components/c1_CaseAnalysis/tests/` are marked `integration`.

## Live LLM (opt-in)

```bash
pytest -m live_llm
```

There is currently **no** dedicated LM Studio smoke-test script in `scripts/`. Retain any future smoke script rather than duplicating it inside pytest. Do not add `live_llm` tests to CI.

## What is mocked

| Boundary | Approach |
|----------|----------|
| Database | Empty `DATABASE_URL` in unit runs; `/health` lifespan patched via `mock_check_connection` |
| C1 intake persistence | `stage2_service.process_intake` mocked in offline intake success test |
| LLM | `get_llm` / `langchain_openai.ChatOpenAI` mocked; no network |

## Gaps (not invented as tests)

- **JWT / protected routes** — `app/core/security.py` is still a TODO; no 401 rejection endpoint exists.
- **LLM timeout/retry wrapper** — not implemented in `get_llm()` yet.
- **Citation validation / relevance scoring** — Stage 4+; not present.
- **C2 civil intake API** — only `GET /api/v1/c2/ping` exists; intake is under **C1** (`/api/v1/c1/intake`).

## GitHub Actions

Workflow: `.github/workflows/backend-ci.yml`  
Required-check name (when enabling branch protection): **`Backend unit tests`**

Local success does **not** prove GitHub Actions passed.
