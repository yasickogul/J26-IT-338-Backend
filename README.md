# Smart Civil Case Analysis Platform — Backend

FastAPI backend (Python 3.11+) for the four legal-analysis components (C1–C4). Uses Neon PostgreSQL + `pgvector`.

## Prerequisites

- Python **3.11+**
- **uv** (Python package manager)
- Neon database (or any Postgres with `pgvector`)

### Install uv (if you don’t have it)

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Confirm:

```bash
uv --version
```

---

## Setup

From this folder (`J26-IT-338-Backend`):

### 1. Create and activate a virtual environment

**macOS / Linux**

```bash
uv venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
uv venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
uv venv
.venv\Scripts\activate.bat
```

> Tip: you can skip activation and prefix commands with `uv run` (see below).

### 2. Install dependencies

```bash
uv pip install -r requirements-dev.txt
```

### 3. Environment file

```bash
cp .env.example .env
```

**Windows**

```powershell
copy .env.example .env
```

Edit `.env` and set at least:

- `DATABASE_URL` — Neon **pooled** URL for the API  
  Format: `postgresql+asyncpg://USER:PASS@HOST-pooler.../DB?ssl=require`  
  (use `ssl=require`, not `sslmode`; remove `channel_binding=require`)
- `DATABASE_URL_DIRECT` — Neon **direct** (non-pooled) URL for Alembic
- `CORS_ORIGINS=http://localhost:3000`
- `JWT_SECRET`, `LLM_*`, `EMBEDDING_*` as needed

---

## Run

With the venv **activated**:

```bash
uvicorn app.main:app --reload --port 8000
```

Or **without** activating (uv runs inside the project venv):

```bash
uv run uvicorn app.main:app --reload --port 8000
```

| URL | Purpose |
|---|---|
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/health | App health |
| http://localhost:8000/health/db | Database health |

On startup you should see: `Database connected (neondb)` (or your DB name).

### Run one component only (optional)

| Component | Command | Port |
|---|---|---|
| C1 | `uv run uvicorn app.components.c1_document_understanding.dev_app:app --reload --port 8001` | 8001 |
| C2 | `uv run uvicorn app.components.c2_case_intelligence.dev_app:app --reload --port 8002` | 8002 |
| C3 | `uv run uvicorn app.components.c3_misinformation.dev_app:app --reload --port 8003` | 8003 |
| C4 | `uv run uvicorn app.components.c_argumentation.dev_app:app --reload --port 8004` | 8004 |

---

## Tests & lint

```bash
uv run pytest -q
uv run ruff check .
```

---

## Project layout (short)

```
app/
  main.py              # full API
  core/                # config, database, llm, …
  shared/              # contracts + retrieval
  components/
    c1_document_understanding/
    c2_case_intelligence/
    c3_misinformation/
    c_argumentation/
```

See `BACKEND_ARCHITECTURE.md` for full design details.
