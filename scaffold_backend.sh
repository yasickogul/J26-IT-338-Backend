#!/usr/bin/env bash
# Generates the backend skeleton described in BACKEND_ARCHITECTURE.md
# Usage:  bash scaffold_backend.sh [target_dir]      (default: backend)
set -euo pipefail

ROOT="${1:-backend}"
COMPONENTS=(
  "c1_document_understanding:c1:8001:C1 – Document Understanding"
  "c2_case_analysis:c2:8002:C2 – Case Analysis"
  "c3_legal_qa:c3:8003:C3 – Legal Q&A"
  "c4_misinformation:c4:8004:C4 – Misinformation Detection"
)

mkdir -p "$ROOT" && cd "$ROOT"

# ---------- top-level dirs ----------
mkdir -p app/core app/shared/{contracts,retrieval,corpus,utils,migrations/versions} \
         app/components scripts data tests/integration

# ---------- top-level files ----------
cat > requirements.txt <<'EOF'
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.7
pydantic-settings>=2.3
sqlalchemy[asyncio]>=2.0
asyncpg>=0.29
alembic>=1.13
pgvector>=0.3
python-multipart>=0.0.9
python-jose[cryptography]>=3.3
passlib[bcrypt]>=1.7
httpx>=0.27
pymupdf>=1.24
langchain>=0.3
langchain-core>=0.3
langgraph>=0.2
# add your LLM provider package, e.g. langchain-anthropic / langchain-openai
# add embedding/NLP libs when needed: sentence-transformers, transformers, torch
EOF

cat > requirements-dev.txt <<'EOF'
-r requirements.txt
pytest>=8
pytest-asyncio>=0.23
ruff>=0.5
EOF

cat > .env.example <<'EOF'
APP_ENV=development
# Neon POOLED string, driver asyncpg. Use ssl=require (NOT sslmode) and remove channel_binding=...
DATABASE_URL=postgresql+asyncpg://USER:PASS@HOST-pooler.neon.tech/DB?ssl=require
# Neon DIRECT (non-pooled) string for Alembic
DATABASE_URL_DIRECT=postgresql+asyncpg://USER:PASS@HOST.neon.tech/DB?ssl=require
JWT_SECRET=change-me
JWT_EXPIRE_MINUTES=60
CORS_ORIGINS=http://localhost:3000
LLM_PROVIDER=anthropic
LLM_MODEL=
LLM_API_KEY=
EMBEDDING_MODEL=
EMBEDDING_DIM=768
FILE_STORAGE=local
FILE_STORAGE_PATH=./data/uploads
EOF

cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.env
data/
.pytest_cache/
.ruff_cache/
EOF

cat > pytest.ini <<'EOF'
[pytest]
asyncio_mode = auto
testpaths = app tests
EOF

cat > CODEOWNERS <<'EOF'
/app/components/c1_document_understanding/   @member1
/app/components/c2_case_analysis/            @member2
/app/components/c3_legal_qa/                 @member3
/app/components/c4_misinformation/           @member4
/app/core/                                   @lead
/app/shared/                                 @lead
/scripts/                                    @lead
EOF

cat > .cursorrules <<'EOF'
You are working on the Smart Civil Case Analysis Platform backend.
Read BACKEND_ARCHITECTURE.md before writing code.
- I work on ONE component (C1/C2/C3/C4). Only edit app/components/<my_component>/.
- Never modify other components, app/core or app/shared unless I explicitly say so.
- Never import another component's internals; use app.shared.* or their public.py only.
- Async FastAPI, Pydantic v2, SQLAlchemy 2.x async, own schema "<cN>" for all tables.
- LLM/embeddings only via app.core.llm / app.core.embeddings.
- Corpus retrieval only via app.shared.retrieval.HybridRetriever.
- All AI outputs grounded in retrieved sources with citations. Add tests for every service function.
EOF

# ---------- app/ ----------
touch app/__init__.py app/core/__init__.py app/shared/__init__.py \
      app/shared/contracts/__init__.py app/shared/retrieval/__init__.py \
      app/shared/corpus/__init__.py app/shared/utils/__init__.py \
      app/components/__init__.py

cat > app/registry.py <<'EOF'
# One line per component. Comment a line out to disable a component.
COMPONENTS = [
    "app.components.c1_document_understanding",
    "app.components.c2_case_analysis",
    "app.components.c3_legal_qa",
    "app.components.c4_misinformation",
]
EOF

cat > app/main.py <<'EOF'
from importlib import import_module

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.registry import COMPONENTS

app = FastAPI(title="Smart Civil Case Analysis Platform", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


for path in COMPONENTS:
    prefix = "/api/v1/" + path.split(".")[-1].split("_")[0]  # -> /api/v1/c1
    try:
        module = import_module(f"{path}.router")
        app.include_router(module.router, prefix=prefix)
    except Exception as e:  # a broken component must not crash the others
        print(f"[WARN] component {path} not loaded: {e}")
EOF

cat > app/core/config.py <<'EOF'
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str = ""
    database_url_direct: str = ""
    jwt_secret: str = "change-me"
    jwt_expire_minutes: int = 60
    cors_origins: str = "http://localhost:3000"

    llm_provider: str = "anthropic"
    llm_model: str = ""
    llm_api_key: str = ""
    embedding_model: str = ""
    embedding_dim: int = 768

    file_storage: str = "local"
    file_storage_path: str = "./data/uploads"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
EOF

cat > app/core/database.py <<'EOF'
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,   # Neon autosuspends idle computes
    pool_size=5,
    max_overflow=5,
) if settings.database_url else None

SessionLocal = async_sessionmaker(engine, expire_on_commit=False) if engine else None


async def get_session() -> AsyncIterator[AsyncSession]:
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL is not set")
    async with SessionLocal() as session:
        yield session
EOF

cat > app/core/llm.py <<'EOF'
from app.core.config import settings


def get_llm(temperature: float = 0.0):
    """Return a LangChain chat model. Edit this ONE place to change provider."""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # pip install langchain-anthropic
        return ChatAnthropic(model=settings.llm_model, api_key=settings.llm_api_key,
                             temperature=temperature)
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI  # pip install langchain-openai
        return ChatOpenAI(model=settings.llm_model, api_key=settings.llm_api_key,
                          temperature=temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
EOF

cat > app/core/embeddings.py <<'EOF'
from app.core.config import settings


def get_embedder():
    """Return a LangChain Embeddings object. ALL components must use this same embedder."""
    from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface sentence-transformers
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
EOF

cat > app/core/security.py <<'EOF'
# TODO (lead): JWT verification dependency, e.g. get_current_user()
EOF
cat > app/core/errors.py <<'EOF'
class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code, self.message, self.status_code = code, message, status_code
EOF
cat > app/core/logging.py <<'EOF'
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
EOF

# ---------- shared contracts ----------
cat > app/shared/contracts/legal_info.py <<'EOF'
from typing import Optional

from pydantic import BaseModel, Field


class LegalEntity(BaseModel):
    text: str
    label: str  # PERSON | ORG | COURT | LOCATION | DATE | CASE_NO | LEGAL_REF
    start: Optional[int] = None
    end: Optional[int] = None


class StructuredLegalInfo(BaseModel):
    """Output of C1; input to C2 / C3 / C4."""
    document_id: Optional[str] = None
    document_type: str = "other"  # judgment | agreement | petition | other
    case_type: Optional[str] = None
    legal_issue: Optional[str] = None
    claims: list[str] = Field(default_factory=list)
    remedy_requested: Optional[str] = None
    parties: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    court: Optional[str] = None
    entities: list[LegalEntity] = Field(default_factory=list)
    raw_text_ref: Optional[str] = None
    confidence: Optional[float] = None
EOF

cat > app/shared/contracts/retrieval.py <<'EOF'
from typing import Optional

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    source_id: str
    title: str
    source_type: str  # act | judgment | regulation | other
    citation: Optional[str] = None
    section: Optional[str] = None
    url: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    citation: SourceCitation
    metadata: dict = Field(default_factory=dict)
EOF

cat > app/shared/contracts/common.py <<'EOF'
from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody
EOF

cat > app/shared/retrieval/service.py <<'EOF'
from app.shared.contracts.retrieval import RetrievedChunk


class HybridRetriever:
    """Keyword (tsvector) + vector (pgvector) search merged with RRF.
    TODO (lead): implement. Components depend on this signature only."""

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: list[str] | None = None,
        filters: dict | None = None,
        alpha: float = 0.5,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError
EOF
cat > app/shared/retrieval/fusion.py <<'EOF'
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
EOF
cat > app/shared/corpus/models.py <<'EOF'
# TODO (lead): SQLAlchemy models for shared.legal_documents and shared.legal_chunks
# (see BACKEND_ARCHITECTURE.md section 5.4)
EOF
touch app/shared/corpus/ingest.py app/shared/utils/text.py app/shared/utils/citations.py

# ---------- components ----------
for entry in "${COMPONENTS[@]}"; do
  IFS=':' read -r folder short port title <<< "$entry"
  base="app/components/$folder"
  mkdir -p "$base"/{migrations/versions,tests,prompts}
  touch "$base/__init__.py" "$base/tests/__init__.py"

  cat > "$base/router.py" <<EOF
from fastapi import APIRouter

router = APIRouter(tags=["$title"])


@router.get("/ping")
async def ping():
    return {"component": "$short", "status": "ok"}
EOF

  cat > "$base/dev_app.py" <<EOF
# Run only this component:
#   uvicorn app.components.$folder.dev_app:app --reload --port $port
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

from .router import router

app = FastAPI(title="$title (dev)")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1/$short")
EOF

  cat > "$base/public.py" <<EOF
"""The ONLY module other components may import from $short.
Expose thin async functions + shared contract types here."""
EOF

  cat > "$base/config.py" <<EOF
# $short-specific settings (prefix env vars with ${short^^}_)
EOF
  touch "$base/schemas.py" "$base/models.py" "$base/repository.py" "$base/service.py"

  cat > "$base/README.md" <<EOF
# $title
Owner: @member
See ../../../BACKEND_ARCHITECTURE.md section for this component.
Run standalone: uvicorn app.components.$folder.dev_app:app --reload --port $port
EOF

  cat > "$base/tests/test_ping.py" <<EOF
from fastapi.testclient import TestClient

from app.components.$folder.dev_app import app


def test_ping():
    r = TestClient(app).get("/api/v1/$short/ping")
    assert r.status_code == 200
EOF
done

# component-specific subfolders
mkdir -p app/components/c1_document_understanding/{pipeline,ml}
mkdir -p app/components/c2_case_analysis/{pipeline,agents,evaluation}
mkdir -p app/components/c3_legal_qa/{rag,agents,memory}
mkdir -p app/components/c4_misinformation/{pipeline,agents,evaluation}
for d in app/components/*/{pipeline,ml,agents,evaluation,rag,memory}; do
  [ -d "$d" ] && touch "$d/__init__.py"
done

cat > tests/conftest.py <<'EOF'
EOF

echo "✅ Skeleton created in: $(pwd)"
echo "Next: python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt"
