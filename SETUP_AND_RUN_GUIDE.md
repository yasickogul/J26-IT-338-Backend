# Setup & Run Guide — Smart Civil Case Analysis Backend

Companion to `BACKEND_ARCHITECTURE.md`. Part A creates the project, Part B runs it, Part C is the daily team workflow.

---

## Part A — Create the Project (one person does this once, then pushes to Git)

### A1. Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.11+ | `python --version` |
| Git | any | `git --version` |
| Neon account | free tier is fine | https://neon.tech |
| Bash | Git Bash or WSL on Windows; built in on macOS/Linux | needed only for the scaffold script |

### A2. Generate the skeleton

Put `scaffold_backend.sh` and `BACKEND_ARCHITECTURE.md` in your repo root, then:

```bash
bash scaffold_backend.sh backend
cp BACKEND_ARCHITECTURE.md backend/
cd backend
```

This creates the full folder tree from the architecture file: `app/core`, `app/shared`, the four `app/components/cN_*` folders (each with `router.py`, `dev_app.py`, `public.py`, `schemas.py`, `models.py`, `service.py`, `repository.py`, `tests/`, etc.), `requirements.txt`, `.env.example`, `.cursorrules`, `CODEOWNERS` and a ping test per component.

### A3. Virtual environment and dependencies

```bash
python -m venv .venv
# macOS/Linux/Git Bash:   source .venv/bin/activate
# Windows PowerShell:     .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Install your LLM provider package when you pick one (e.g. `pip install langchain-anthropic` or `langchain-openai`). Heavy ML packages (`torch`, `transformers`, `sentence-transformers`, `langchain-huggingface`) are best installed by the member who needs them (C1, C2, C4) to keep everyone's setup light.

### A4. Set up Neon

1. Create a Neon project (pick the region closest to you). The default branch is your `main`.
2. Open the **SQL Editor** and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE SCHEMA IF NOT EXISTS shared;
   CREATE SCHEMA IF NOT EXISTS c1;
   CREATE SCHEMA IF NOT EXISTS c2;
   CREATE SCHEMA IF NOT EXISTS c3;
   CREATE SCHEMA IF NOT EXISTS c4;
   ```
3. **Create one Neon branch per member** (Branches → New branch, e.g. `dev-member1` … `dev-member4`). Each branch is a copy-on-write copy of `main`, so experiments never collide.
4. For each branch, copy the connection strings from **Connect**:
   - **Pooled** string (host contains `-pooler`) → `DATABASE_URL` (used by the API)
   - **Direct** string → `DATABASE_URL_DIRECT` (used by Alembic)
5. Convert the strings for the async driver:
   - Change the start to `postgresql+asyncpg://`
   - Replace `sslmode=require` with `ssl=require`
   - **Delete** `&channel_binding=require` (asyncpg rejects it)

   ```
   DATABASE_URL=postgresql+asyncpg://user:pass@ep-xxx-pooler.region.aws.neon.tech/neondb?ssl=require
   ```

### A5. Environment file

```bash
cp .env.example .env      # never commit .env
```

Fill in `DATABASE_URL`, `DATABASE_URL_DIRECT`, `JWT_SECRET`, `LLM_*`, `EMBEDDING_*`. Each member has their own `.env` pointing at their own Neon branch.

### A6. Set up Alembic (multi-branch migrations)

```bash
alembic init -t async alembic
```

Edit `alembic.ini`:

```ini
script_location = alembic
version_locations = app/components/c1_document_understanding/migrations/versions app/components/c2_case_analysis/migrations/versions app/components/c3_legal_qa/migrations/versions app/components/c4_misinformation/migrations/versions app/shared/migrations/versions
```

Replace `alembic/env.py`'s metadata and URL parts with:

```python
from alembic import context
from app.core.config import settings

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url_direct)

# Import each component's Base as soon as that component has models.
# Each component defines:  Base = declarative_base(metadata=MetaData(schema="cN"))
from app.components.c1_document_understanding.models import Base as C1Base   # noqa
from app.components.c2_case_analysis.models import Base as C2Base             # noqa
from app.components.c3_legal_qa.models import Base as C3Base                  # noqa
from app.components.c4_misinformation.models import Base as C4Base            # noqa

target_metadata = [C1Base.metadata, C2Base.metadata, C3Base.metadata, C4Base.metadata]

# Lets each member autogenerate only their own schema:  alembic -x schema=c2 revision --autogenerate ...
SCHEMA = context.get_x_argument(as_dictionary=True).get("schema")

def include_name(name, type_, parent_names):
    if type_ == "schema":
        return SCHEMA is None or name == SCHEMA
    return True
```

and pass `include_schemas=True, include_name=include_name` inside `context.configure(...)` in both the offline and online functions.

> Tip: until a component has a `models.py` with a `Base`, comment out its import line. Each member adds their own line when they write their first model.

**Each member's migration workflow** (example for C2):

```bash
# first migration for the component (starts its own branch)
alembic -x schema=c2 revision --autogenerate -m "create c2 tables" \
  --head=base --branch-label=c2 \
  --version-path=app/components/c2_case_analysis/migrations/versions

# later migrations
alembic -x schema=c2 revision --autogenerate -m "add whatif_runs" --head=c2@head

# apply only your branch
alembic upgrade c2@head
```

### A7. Create the shared corpus tables

Whoever owns `shared/` writes the SQLAlchemy models in `app/shared/corpus/models.py` (`legal_documents`, `legal_chunks` — SQL in the architecture file §5.4) and creates the first migration with `--branch-label=shared` and `--version-path=app/shared/migrations/versions`. Set the vector dimension to match `EMBEDDING_DIM`.

### A8. First commit

```bash
cd ..            # repo root
git init         # skip if the repo exists
git add . && git commit -m "chore: scaffold backend"
git branch develop && git push -u origin main develop
```

Protect `main` and `develop` on GitHub (require PR + 1 review).

---

## Part B — Run the Project

### B1. Run the whole API

From `backend/` with the venv active:

```bash
uvicorn app.main:app --reload --port 8000
```

- API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Component ping: http://localhost:8000/api/v1/c1/ping (also `c2`, `c3`, `c4`)

If a component fails to import, the app still starts and prints `[WARN] component ... not loaded`.

### B2. Run only your component (recommended while developing)

| Component | Command | Port |
|---|---|---|
| C1 | `uvicorn app.components.c1_document_understanding.dev_app:app --reload --port 8001` | 8001 |
| C2 | `uvicorn app.components.c2_case_analysis.dev_app:app --reload --port 8002` | 8002 |
| C3 | `uvicorn app.components.c3_legal_qa.dev_app:app --reload --port 8003` | 8003 |
| C4 | `uvicorn app.components.c4_misinformation.dev_app:app --reload --port 8004` | 8004 |

Swagger for each is at `http://localhost:<port>/docs`.

### B3. Run tests and lint

```bash
pytest -q                                          # everything
pytest app/components/c2_case_analysis -q          # only your component
ruff check . && ruff format .
```

### B4. Load data (once the shared layer is ready)

```bash
python -m scripts.ingest_judgments --path data/judgments     # Court of Appeal scraper output
python -m scripts.ingest_acts --path data/acts
python -m scripts.build_embeddings                            # fill legal_chunks.embedding
```

Start with a **small sample (50–100 documents)** so retrieval can be tested quickly, then ingest the full corpus.

### B5. Connect the Next.js frontend

In the frontend `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Call endpoints like `fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/c3/conversations`, ...)`. CORS already allows `http://localhost:3000` through `CORS_ORIGINS` in the backend `.env`. For streaming (C3), read the SSE endpoint with `fetch` + `ReadableStream` (or `EventSource` for GET-based streams).

### B6. Optional: Docker

Only needed if you want an environment without Neon (offline dev):

```yaml
# docker-compose.yml
services:
  db:
    image: pgvector/pgvector:pg16
    environment: { POSTGRES_USER: app, POSTGRES_PASSWORD: app, POSTGRES_DB: legal }
    ports: ["5432:5432"]
```

Then `DATABASE_URL=postgresql+asyncpg://app:app@localhost:5432/legal`.

---

## Part C — Daily Team Workflow

1. `git checkout develop && git pull`
2. `git checkout -b c2/relevance-ranker` (branch prefix = your component)
3. Open Cursor, and start prompts with: *"Read BACKEND_ARCHITECTURE.md. I'm working on C2 only…"* (`.cursorrules` already enforces the boundaries).
4. Code inside **your component folder only**; run your `dev_app`; write tests.
5. `alembic upgrade cN@head` on your own Neon branch.
6. `ruff check . && pytest app/components/<yours> -q`
7. Commit as `feat(c2): ...`, push, open a PR into `develop`. PRs touching another component's folder should be rejected. Changes to `core/` or `shared/` need all four members' review.

### Suggested first tasks per member

| Member | First 3 tasks |
|---|---|
| **Lead / everyone (Week 0)** | Implement `HybridRetriever`, shared corpus tables + migration, ingest sample data, JWT dependency in `core/security.py` |
| **C1** | `POST /documents` upload, `pdf_extractor.py` (PyMuPDF), `StructuredLegalInfo` extraction (LLM-based baseline, then classifier/NER models) |
| **C2** | Fact extraction + query builder, `ranker.py` with the weighted percentage score, explanation chain with structured output |
| **C3** | Conversations/messages tables, RAG chain with citations, SSE streaming endpoint |
| **C4** | Claim extractor, verifier with three verdicts (default to Insufficient Evidence), `/verifications` endpoints |

Until teammates finish, **develop against mocks** of their `public.py` and the shared contracts.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: app` | Run commands from the `backend/` folder, not from inside `app/` |
| `asyncpg ... unexpected keyword 'sslmode'` / `channel_binding` | Use `ssl=require`; remove `channel_binding=require` |
| `type "vector" does not exist` | Run `CREATE EXTENSION IF NOT EXISTS vector;` on that Neon branch |
| First request slow / connection dropped | Neon compute was suspended; `pool_pre_ping=True` is already set, retry once |
| Alembic can't find models/tables | Import the component's `Base` in `alembic/env.py`; pass `-x schema=cN` |
| `Multiple heads` error | Use branch targets: `alembic upgrade c2@head`, not `head` |
| Vector size mismatch | `EMBEDDING_DIM` must equal the `VECTOR(n)` column size and the embedding model's output |
| Component missing from `/docs` | Check the `[WARN] component ... not loaded` message in the terminal |
| Windows can't run `.sh` | Use Git Bash or WSL |

---

## Quick Command Cheat Sheet

```bash
# setup
python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
cp .env.example .env

# run
uvicorn app.main:app --reload --port 8000                                   # full API
uvicorn app.components.c2_case_analysis.dev_app:app --reload --port 8002    # one component

# database
alembic upgrade c2@head                     # apply your component's migrations
alembic -x schema=c2 revision --autogenerate -m "msg" --head=c2@head

# quality
pytest app/components/c2_case_analysis -q
ruff check . && ruff format .
```
