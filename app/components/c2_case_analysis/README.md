# C2 Civil Case Intelligence

The imported backend is contained in this C2 component. The original combined source folder has been removed; historical documentation is in `docs/ORIGINAL_PROJECT_README.md`. The frontend is in the separate
`J26-IT-338-Frontend` repository on branch `kavy`; there is no frontend in this backend component.

## Run from the backend repository root (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r app/components/c2_case_analysis/requirements.txt
# For a fresh clone only: copy .env.example to .env, then fill in your service settings.
Copy-Item app/components/c2_case_analysis/.env.example app/components/c2_case_analysis/.env
.\.venv\Scripts\python.exe -m uvicorn app.components.c2_case_analysis.dev_app:app --reload --port 8000
```

The supplied local `.env` was copied during import and is ignored by Git. Do not
overwrite it with the example when running this existing working copy.

API docs: http://127.0.0.1:8000/docs
Readiness: http://127.0.0.1:8000/api/v1/c2/health
Wait for `warmup: ready` before analysis. Startup loads OCR/embedding models and
connects to external services. Analysis requires configured credentials.

## API

All endpoints are under `/api/v1/c2`: `GET /health`, `POST /analyze`,
`GET /jobs/{id}`, `DELETE /jobs/{id}`, `GET /jobs/{id}/extraction`,
`GET /jobs/{id}/result`, `GET /jobs/{id}/report.pdf`, and `GET /jobs/{id}/report.md`.
The frontend proxies these requests to port 8000 during development.

The team application can import `router` from this component's `router.py` and
mount it with prefix `/api/v1/c2`. It must also manage startup warm-up and CORS.
The ingestion script is available as
`python -m app.components.c2_case_analysis.scripts.build_knowledge_base` from the
repository root. It writes to Qdrant; it is not needed for normal startup when
the collection already exists.

## Architecture status

This change places the supplied implementation inside the C2 module and adjusts
imports, local data paths, and route prefixes. Its analysis pipeline is retained.
It still uses Qdrant, direct OpenRouter calls, and in-memory jobs. Shared
Neon/pgvector retrieval, LangChain/LangGraph integration, persistence, auth, the
`public.py` contract, and the planned `/analyses` API remain future integration
work described in `BACKEND_ARCHITECTURE.md`.
