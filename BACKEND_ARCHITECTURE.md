# Smart Civil Case Analysis Platform — Backend Architecture Guide

> **For Cursor / AI assistants:** This file is the source of truth for the backend. Before generating any code, read this file fully. Follow the folder structure, ownership rules, and contracts exactly. Never edit files outside the component you were asked to work on.

---

## 1. Project Summary

An AI-powered legal assistance platform for **Sri Lankan civil law**. Users submit a legal problem or upload legal documents; the system understands the input, retrieves relevant Sri Lankan legal sources and precedents, analyses the case, answers legal questions, verifies legal claims, and returns **structured, explainable, source-cited** results.

**Stack**

| Layer | Technology |
|---|---|
| Frontend (done) | Next.js |
| Backend API | Python 3.11+, FastAPI (async) |
| Database | PostgreSQL on **Neon** + `pgvector` extension |
| Keyword retrieval | PostgreSQL full-text search (`tsvector`, BM25-style ranking) or `rank_bm25` |
| Vector retrieval | `pgvector` (FAISS optional, only for local experiments) |
| AI orchestration / agents | **LangChain** + **LangGraph** (for multi-step agents) |
| Legal NLP | Legal-BERT (classification / NER / features), sentence-embedding model for retrieval |
| PDF processing | PyMuPDF |
| ORM / migrations | SQLAlchemy 2.x (async) + Alembic |
| Validation | Pydantic v2 |

**High-level flow**

```
User Input → Document/Claim Understanding (C1) → Legal Retrieval → Civil Case Analysis (C2)
          → Legal Q&A (C3) / Claim Verification (C4) → Explainable Results
```

### The four components

| ID | Component | Owner | Purpose |
|---|---|---|---|
| C1 | Legal Document Classification & Information Extraction | Member 1 | Extract text from uploads, classify document type, run NER, output structured legal info |
| C2 | Explainable Civil Case Analysis | Member 2 | Retrieve and rank relevant precedents, percentage relevance score, explanation, evidence gaps, what-if |
| C3 | Citizen Legal Q&A Assistant | Member 3 | RAG chatbot answering legal questions with citations |
| C4 | Legal Misinformation Detection | Member 4 | Verify a legal claim: Supported / Contradicted / Insufficient Evidence |

*(Replace "Member N" with real names/GitHub handles in `CODEOWNERS`.)*

---

## 2. Architectural Style: Modular Monolith

One FastAPI application, but each component is a **self-contained module** that mounts its own router. This gives:

- One deployment and one Neon database (simple for a 4-person team).
- Strict isolation: each member works only inside `app/components/cX_*/`.
- Each component can also run **standalone** for development (`dev_app.py`), so nobody waits for the others.

```
                      ┌───────────────────────────┐
                      │   Next.js Frontend (done) │
                      └─────────────┬─────────────┘
                                    │ HTTPS / JSON (+ SSE for streaming)
                      ┌─────────────▼─────────────┐
                      │   FastAPI  (app/main.py)   │
                      │  auth · CORS · logging     │
                      └──┬─────────┬─────────┬─────┬┘
             /api/v1/c1  │  /c2    │  /c3    │ /c4 │
          ┌──────────────▼┐ ┌──────▼──────┐ ┌▼─────▼───────┐ ┌────────────┐
          │ C1 Document   │ │ C2 Case     │ │ C3 Legal Q&A │ │ C4 Misinfo │
          │ Understanding │ │ Analysis    │ │              │ │ Detection  │
          └──────┬────────┘ └──────┬──────┘ └──────┬───────┘ └─────┬──────┘
                 │                 │               │               │
        ┌────────▼─────────────────▼───────────────▼───────────────▼───────┐
        │  app/core  (config, db, llm, embeddings)  +  app/shared          │
        │  (contracts, retrieval service, citation utils)                  │
        └────────────────────────────┬─────────────────────────────────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │ Neon PostgreSQL + pgvector      │
                     │ schemas: shared · c1 · c2 · c3 · c4 │
                     └────────────────────────────────┘
```

---

## 3. Folder Structure

```
backend/
├── README.md
├── BACKEND_ARCHITECTURE.md          # this file
├── .cursorrules                     # or .cursor/rules/*.mdc  (see §14)
├── CODEOWNERS                       # per-folder ownership (see §12)
├── pyproject.toml                   # deps (or requirements.txt)
├── .env.example
├── alembic.ini                      # multi-branch, see §7
├── docker-compose.yml               # optional local Postgres+pgvector
│
├── app/
│   ├── main.py                      # builds FastAPI app, auto-mounts component routers
│   ├── registry.py                  # list of components to mount (1 line per component)
│   │
│   ├── core/                        # 🔒 SHARED INFRA — change only via PR + team review
│   │   ├── config.py                # pydantic-settings, env vars
│   │   ├── database.py              # async engine, session factory (Neon)
│   │   ├── security.py              # JWT auth, dependencies
│   │   ├── logging.py
│   │   ├── errors.py                # common exception classes + handlers
│   │   ├── llm.py                   # get_llm() factory (LangChain chat model)
│   │   └── embeddings.py            # get_embedder() factory
│   │
│   ├── shared/                      # 🔒 CONTRACTS + COMMON SERVICES
│   │   ├── contracts/               # Pydantic models used BETWEEN components
│   │   │   ├── legal_info.py        # StructuredLegalInfo (C1 output)
│   │   │   ├── retrieval.py         # RetrievedChunk, SourceCitation
│   │   │   └── common.py            # Pagination, ErrorResponse, etc.
│   │   ├── retrieval/               # hybrid retrieval over the legal corpus (BM25 + pgvector)
│   │   │   ├── service.py           # HybridRetriever
│   │   │   └── fusion.py            # Reciprocal Rank Fusion
│   │   ├── corpus/                  # legal corpus tables + ingestion
│   │   │   ├── models.py            # shared.legal_documents, shared.legal_chunks
│   │   │   └── ingest.py
│   │   └── utils/
│   │       ├── text.py
│   │       └── citations.py
│   │
│   └── components/
│       ├── c1_document_understanding/     # 👤 Member 1 ONLY
│       │   ├── __init__.py
│       │   ├── public.py                  # ONLY thing other components may import
│       │   ├── router.py                  # FastAPI routes  → /api/v1/c1
│       │   ├── dev_app.py                 # standalone runner for this component
│       │   ├── config.py                  # component-specific settings
│       │   ├── schemas.py                 # request/response models
│       │   ├── models.py                  # SQLAlchemy models (schema "c1")
│       │   ├── repository.py              # DB access
│       │   ├── service.py                 # orchestrates the pipeline
│       │   ├── pipeline/
│       │   │   ├── pdf_extractor.py       # PyMuPDF
│       │   │   ├── preprocessor.py        # cleaning, sentence split, language detect
│       │   │   ├── classifier.py          # document type classification
│       │   │   ├── ner.py                 # entities: parties, courts, dates, case no.
│       │   │   └── extractor.py           # builds StructuredLegalInfo
│       │   ├── ml/                        # model weights loaders, training scripts
│       │   ├── prompts/                   # LLM prompts (if LLM-assisted extraction)
│       │   ├── migrations/versions/       # Alembic versions (branch "c1")
│       │   ├── tests/
│       │   └── README.md
│       │
│       ├── c2_case_analysis/              # 👤 Member 2 ONLY
│       │   ├── __init__.py
│       │   ├── public.py
│       │   ├── router.py                  # → /api/v1/c2
│       │   ├── dev_app.py
│       │   ├── config.py                  # scoring weights, top_k, thresholds
│       │   ├── schemas.py
│       │   ├── models.py                  # schema "c2"
│       │   ├── repository.py
│       │   ├── service.py
│       │   ├── pipeline/
│       │   │   ├── fact_extractor.py      # claim/fact extraction from current case
│       │   │   ├── query_builder.py       # query representation
│       │   │   ├── retriever.py           # uses shared.retrieval + case-level index
│       │   │   ├── ranker.py              # relevance percentage scoring
│       │   │   ├── comparator.py          # fact & evidence comparison
│       │   │   ├── evidence_gap.py
│       │   │   └── whatif.py
│       │   ├── agents/
│       │   │   ├── explanation_agent.py   # LangChain: explains why relevant
│       │   │   └── graph.py               # LangGraph workflow
│       │   ├── prompts/
│       │   ├── evaluation/                # precision/recall/F1, retrieval metrics
│       │   ├── migrations/versions/       # branch "c2"
│       │   ├── tests/
│       │   └── README.md
│       │
│       ├── c3_legal_qa/                   # 👤 Member 3 ONLY
│       │   ├── __init__.py
│       │   ├── public.py
│       │   ├── router.py                  # → /api/v1/c3  (supports SSE streaming)
│       │   ├── dev_app.py
│       │   ├── config.py
│       │   ├── schemas.py
│       │   ├── models.py                  # schema "c3" (conversations, messages)
│       │   ├── repository.py
│       │   ├── service.py
│       │   ├── rag/
│       │   │   ├── query_rewriter.py      # follow-up question → standalone question
│       │   │   ├── retriever.py           # uses shared.retrieval
│       │   │   ├── context_builder.py
│       │   │   └── answer_chain.py        # LangChain RAG chain
│       │   ├── agents/
│       │   │   └── qa_agent.py            # LangGraph/LangChain agent with tools
│       │   ├── memory/                    # conversation memory
│       │   ├── prompts/
│       │   ├── migrations/versions/       # branch "c3"
│       │   ├── tests/
│       │   └── README.md
│       │
│       └── c4_misinformation/             # 👤 Member 4 ONLY
│           ├── __init__.py
│           ├── public.py
│           ├── router.py                  # → /api/v1/c4
│           ├── dev_app.py
│           ├── config.py
│           ├── schemas.py
│           ├── models.py                  # schema "c4" (claims, verifications)
│           ├── repository.py
│           ├── service.py
│           ├── pipeline/
│           │   ├── claim_extractor.py
│           │   ├── concept_identifier.py
│           │   ├── source_retriever.py    # uses shared.retrieval
│           │   ├── comparator.py          # semantic / NLI evidence comparison
│           │   └── verifier.py            # Supported / Contradicted / Insufficient
│           ├── agents/
│           │   └── verification_agent.py
│           ├── prompts/
│           ├── evaluation/
│           ├── migrations/versions/       # branch "c4"
│           ├── tests/
│           └── README.md
│
├── scripts/                         # data ingestion, embedding backfill, etc.
│   ├── ingest_acts.py
│   ├── ingest_judgments.py
│   └── build_embeddings.py
├── data/                            # local raw/processed legal data (gitignored if large)
└── tests/
    ├── conftest.py
    └── integration/                 # cross-component tests (added at integration phase)
```

---

## 4. Isolation Rules (critical)

1. **Own your folder only.** Member N edits only `app/components/cN_*/`. Never edit another component's folder.
2. **No cross-imports of internals.** A component must NOT do `from app.components.c2_case_analysis.service import ...`. Allowed imports:
   - `app.core.*`
   - `app.shared.*`
   - Another component's `public.py` (a thin facade; only functions/types listed there).
3. **Cross-component data goes through contracts** in `app/shared/contracts/`. Example: C1's output is `StructuredLegalInfo`; C2, C3, and C4 consume that type, not C1's internal models.
4. **Changes to `core/` or `shared/`** need a PR reviewed by all 4 members (or the team lead). If you need something new, open a PR that adds it — don't fork a private copy.
5. **Database ownership:** each component has its own Postgres schema (`c1`, `c2`, `c3`, `c4`) and only writes to its own. The `shared` schema (legal corpus) is **read-only** for components; only `scripts/` ingestion writes to it.
6. **Migrations:** each component keeps its own Alembic branch (see §7). Never edit another component's migrations.
7. **Dependencies:** add a new package in a small dedicated PR (or a component-level extras group in `pyproject.toml`, e.g. `[project.optional-dependencies] c2 = [...]`) to avoid merge conflicts.
8. **Mocks first:** until another component is ready, develop against its **contract** with a mock (`tests/fixtures`). Never block on a teammate.

---

## 5. Shared Layer

### 5.1 `app/core`

- **config.py** — `Settings(BaseSettings)` reading `.env`. Includes `DATABASE_URL`, `LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `JWT_SECRET`, `CORS_ORIGINS`.
- **database.py** — async SQLAlchemy engine + `get_session()` dependency. Notes for Neon:
  - Use the driver `postgresql+asyncpg://`.
  - Use Neon's **pooled** connection string for the API; use the **direct** (non-pooled) string for Alembic migrations.
  - Enable pgvector once: `CREATE EXTENSION IF NOT EXISTS vector;`
  - Neon autosuspends: enable `pool_pre_ping=True` and keep pool sizes small.
- **llm.py** — `get_llm(temperature=0.0)` returns a LangChain chat model; every component obtains models here so the provider can be swapped in one place.
- **embeddings.py** — `get_embedder()` returns a LangChain `Embeddings` object. **All components must use the same embedder** as the corpus (vectors are only comparable within one model). Record the model name + dimension in config.

### 5.2 `app/shared/contracts`

The only types shared across components. Keep them small and versioned.

```python
# app/shared/contracts/legal_info.py
from pydantic import BaseModel, Field
from typing import Optional

class LegalEntity(BaseModel):
    text: str
    label: str            # PERSON | ORG | COURT | LOCATION | DATE | CASE_NO | LEGAL_REF
    start: Optional[int] = None
    end: Optional[int] = None

class StructuredLegalInfo(BaseModel):
    """Output of C1; input to C2 / C3 / C4."""
    document_id: Optional[str] = None
    document_type: str                       # judgment | agreement | petition | other
    case_type: Optional[str] = None          # e.g. Employment
    legal_issue: Optional[str] = None        # e.g. Wrongful Termination
    claims: list[str] = Field(default_factory=list)
    remedy_requested: Optional[str] = None
    parties: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    court: Optional[str] = None
    entities: list[LegalEntity] = Field(default_factory=list)
    raw_text_ref: Optional[str] = None       # pointer, not the full text
    confidence: Optional[float] = None
```

```python
# app/shared/contracts/retrieval.py
class SourceCitation(BaseModel):
    source_id: str
    title: str
    source_type: str              # act | judgment | regulation | other
    citation: Optional[str] = None
    section: Optional[str] = None
    url: Optional[str] = None

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    citation: SourceCitation
    metadata: dict = {}
```

### 5.3 `app/shared/retrieval` — Hybrid retrieval service

Used by C2, C3, C4 so nobody re-implements retrieval.

```python
class HybridRetriever:
    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: list[str] | None = None,   # filter: ["judgment"], ["act"], ...
        filters: dict | None = None,             # court, year, case_type
        alpha: float = 0.5,                      # keyword vs vector balance
    ) -> list[RetrievedChunk]: ...
```

Implementation: run keyword search (Postgres `ts_rank_cd` on a `tsvector` column) and vector search (`embedding <=> query_vec`) in parallel, then merge with **Reciprocal Rank Fusion**. Optional re-ranking step (cross-encoder) can be added later without changing the interface.

### 5.4 Shared legal corpus (schema `shared`)

```sql
CREATE SCHEMA IF NOT EXISTS shared;

CREATE TABLE shared.legal_documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_type TEXT NOT NULL,            -- act | judgment | regulation
  title TEXT NOT NULL,
  citation TEXT,                        -- case number / act number
  court TEXT,
  case_type TEXT,
  decision_date DATE,
  language TEXT DEFAULT 'en',
  source_url TEXT,
  full_text TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE shared.legal_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES shared.legal_documents(id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  section TEXT,                         -- e.g. "Section 12", "Held:" 
  text TEXT NOT NULL,
  tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  embedding VECTOR(768),                -- dimension = EMBEDDING_DIM
  metadata JSONB DEFAULT '{}'
);

CREATE INDEX ON shared.legal_chunks USING GIN (tsv);
CREATE INDEX ON shared.legal_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON shared.legal_chunks (document_id);
```

Ingestion (`scripts/ingest_*.py`): parse → clean → chunk (by section/paragraph, ~300–500 tokens with overlap) → embed → insert. The Court of Appeal scraper output can feed `ingest_judgments.py`.

---

## 6. Component Designs

### 6.1 C1 — Legal Document Classification & Information Extraction

**Pipeline**

```
Upload (PDF/text) → PyMuPDF text extraction → preprocessing (clean, sentence split,
language detect, normalize) → document classification → NER → information extraction
→ StructuredLegalInfo → saved in c1 schema → returned / consumed by C2, C3, C4
```

**Endpoints (`/api/v1/c1`)**

| Method | Path | Description |
|---|---|---|
| POST | `/documents` | Upload file (multipart). Returns `document_id` + status |
| GET | `/documents/{id}` | Metadata + processing status |
| GET | `/documents/{id}/structured` | `StructuredLegalInfo` |
| GET | `/documents/{id}/text` | Extracted text |
| POST | `/extract-text` | Free-text input (no file) → `StructuredLegalInfo` |
| DELETE | `/documents/{id}` | Delete document + derived data |

**Tables (schema `c1`)**: `documents` (id, filename, mime, storage_path, status, created_at), `extractions` (document_id, doc_type, structured_json JSONB, confidence, model_version), `entities` (document_id, text, label, start, end).

**Notes**
- Long-running work (OCR, model inference) → FastAPI `BackgroundTasks` first; move to a task queue (Celery/RQ/Arq) only if needed. Status field: `queued → processing → done | failed`.
- Scanned PDFs need OCR (e.g., Tesseract) as a fallback when PyMuPDF returns little text.
- Classification: fine-tuned Legal-BERT (or LLM zero-shot as baseline). NER: fine-tuned model, regex for case numbers/dates, LLM-assisted for claims/issue/remedy.
- Store uploaded files in object storage (S3/Cloudflare R2/Supabase Storage) or local disk in dev; save only the path in DB.

**`public.py` exposes:** `async def get_structured_info(document_id) -> StructuredLegalInfo`, `async def analyse_text(text) -> StructuredLegalInfo`.

---

### 6.2 C2 — Explainable Civil Case Analysis

**Pipeline**

```
Current case (form or StructuredLegalInfo) → claim/fact extraction → query representation
→ hybrid retrieval (BM25 + pgvector) → candidate cases → legal relevance ranking
→ relevance percentage → fact & evidence comparison → explanation generation
→ evidence-gap analysis → (optional) what-if analysis
```

**Endpoints (`/api/v1/c2`)**

| Method | Path | Description |
|---|---|---|
| POST | `/analyses` | Submit case (structured input or `document_id`) → `analysis_id` |
| GET | `/analyses/{id}` | Status + results |
| GET | `/analyses/{id}/cases` | Ranked relevant cases with percentage + factor breakdown |
| GET | `/analyses/{id}/cases/{case_id}/explanation` | Similarities, differences, reasoning |
| GET | `/analyses/{id}/evidence-gaps` | Missing evidence vs. precedents |
| POST | `/analyses/{id}/what-if` | Change a legal fact → re-run and compare (**clearly labelled hypothetical**) |
| GET | `/analyses` | User's history |

**Relevance scoring (percentage-based relevance score — NOT a probability)**

```python
score = 100 * (
    w_issue    * issue_sim +
    w_fact     * fact_sim +
    w_claim    * claim_sim +
    w_evidence * evidence_sim +
    w_remedy   * remedy_sim +
    w_court    * court_precedent_weight
)
# each *_sim in [0, 1]; weights sum to 1; defined in c2/config.py
```

- Each factor similarity = embedding cosine similarity (Legal-BERT/sentence embedding) on the matching extracted field, optionally combined with rule-based matching (same case type, same statute cited).
- Also return a qualitative label per factor (High / Medium / Low) using thresholds in config.
- **Weights are initial guesses**; the research must tune and validate them (e.g., against lawyer-labelled relevance judgments; grid search / learning-to-rank). Keep weights in config and store the `scoring_version` on every result for reproducibility.
- Always describe the output as a *percentage-based relevance score*.

**Explanation (LangChain agent / chain)**
- Input: current case facts + retrieved precedent chunks + factor scores.
- Output (structured, Pydantic): `similarities[]`, `differences[]`, `factor_reasons{}`, `summary`.
- The LLM must only use provided evidence; every statement links to a chunk/citation. Use `with_structured_output(...)`.
- LangGraph flow: `extract_facts → retrieve → rank → compare → explain → evidence_gap` (each a node; state is a typed dict).

**What-if:** clone the case features, apply the modified fact, re-run retrieval + ranking, return a diff (cases entering/leaving the list, score changes). Response is flagged `"mode": "hypothetical"` and never stored as the actual case's result.

**Tables (schema `c2`)**: `analyses`, `analysis_cases` (analysis_id, case_id, score, factor_scores JSONB, scoring_version), `explanations`, `evidence_gaps`, `whatif_runs`.

**Evaluation (`evaluation/`)**: Precision@k, Recall@k, F1, MRR/nDCG, confusion matrix on a labelled relevance set.

**`public.py` exposes:** `async def analyse_case(info: StructuredLegalInfo) -> AnalysisResult`.

---

### 6.3 C3 — Citizen Legal Q&A Assistant

**Pipeline**

```
User question → (rewrite using chat history) → hybrid retrieval (acts/judgments)
→ context construction (dedupe, order, token budget) → LLM → grounded answer
→ citations/sources → saved to conversation
```

**Endpoints (`/api/v1/c3`)**

| Method | Path | Description |
|---|---|---|
| POST | `/conversations` | Create conversation |
| GET | `/conversations` | List user's conversations |
| GET | `/conversations/{id}` | Messages |
| POST | `/conversations/{id}/messages` | Ask question → answer + sources (JSON) |
| POST | `/conversations/{id}/messages/stream` | Same, streamed via **SSE** |
| DELETE | `/conversations/{id}` | Delete |
| POST | `/conversations/{id}/context` | Attach a C1 `document_id` or C2 `analysis_id` as context |

**Rules for the answer chain**
- Answer **only** from retrieved context; if the context is insufficient, say so instead of guessing.
- Every answer includes `sources: SourceCitation[]`.
- Add a "not legal advice" disclaimer field in the response.
- Optional agent tools: `search_legal_corpus`, `get_document_info` (via C1 `public.py`), `get_case_analysis` (via C2 `public.py`).

**Tables (schema `c3`)**: `conversations`, `messages` (role, content, sources JSONB, created_at), `feedback` (message_id, rating, comment).

**`public.py` exposes:** `async def answer_question(question, history=None) -> Answer`.

---

### 6.4 C4 — Legal Misinformation Detection

**Pipeline**

```
Legal claim → claim extraction (atomic claims) → legal concept identification
→ relevant source retrieval → semantic/evidence comparison (NLI or LLM judge)
→ verdict per claim: Supported | Contradicted | Insufficient Evidence
→ explanation + sources
```

**Endpoints (`/api/v1/c4`)**

| Method | Path | Description |
|---|---|---|
| POST | `/verifications` | Submit text/claim → `verification_id` |
| GET | `/verifications/{id}` | Verdict(s) per atomic claim, explanation, evidence |
| POST | `/verify-answer` | Verify an LLM-generated answer (e.g., from C3) against sources |
| GET | `/verifications` | History |

**Response shape**

```json
{
  "claims": [
    {
      "claim": "…",
      "verdict": "Supported | Contradicted | Insufficient Evidence",
      "confidence": 0.82,
      "explanation": "…",
      "evidence": [ { "citation": {…}, "excerpt": "…", "stance": "supports|contradicts" } ]
    }
  ]
}
```

**Rules**
- Default to **Insufficient Evidence** when retrieval confidence is low. Never mark Supported without at least one cited passage.
- Combine an NLI/cross-encoder model (entailment / contradiction / neutral) with an LLM judge for the explanation.
- Evaluate with a labelled claim set (accuracy, per-class precision/recall, confusion matrix).

**Tables (schema `c4`)**: `verifications`, `claims`, `claim_evidence`.

**`public.py` exposes:** `async def verify_claims(text: str) -> VerificationResult` (C3 may call this to check its own answers).

---

## 7. Database & Migrations (Neon)

- **One Neon database**, five schemas: `shared`, `c1`, `c2`, `c3`, `c4`. Each component's SQLAlchemy models set `__table_args__ = {"schema": "cN"}` and use their **own** `DeclarativeBase`/`MetaData`.
- **Alembic multi-branch:** in `alembic.ini` list every `version_locations`; each migration file declares `branch_labels = ("c1",)` (etc.). Members run `alembic upgrade c1@head` for their own branch only.
  ```
  version_locations = app/components/c1_document_understanding/migrations/versions
                      app/components/c2_case_analysis/migrations/versions
                      app/components/c3_legal_qa/migrations/versions
                      app/components/c4_misinformation/migrations/versions
                      app/shared/migrations/versions
  ```
- **Dev isolation on Neon:** use **Neon branches** — every member gets their own database branch (copy-on-write) so experiments never break each other's data. `main` branch = integration/staging.
- Cross-component foreign keys are **not allowed**. Reference other components by plain UUID columns (no FK) and resolve through `public.py`.

---

## 8. Component Registration & App Bootstrap

```python
# app/registry.py  — one line per component
COMPONENTS = [
    "app.components.c1_document_understanding",
    "app.components.c2_case_analysis",
    "app.components.c3_legal_qa",
    "app.components.c4_misinformation",
]
```

```python
# app/main.py
from importlib import import_module
from fastapi import FastAPI
from app.registry import COMPONENTS

app = FastAPI(title="Smart Civil Case Analysis Platform", version="0.1.0")

for path in COMPONENTS:
    try:
        module = import_module(f"{path}.router")
        app.include_router(module.router, prefix=f"/api/v1/{path.split('.')[-1].split('_')[0]}")
    except ImportError as e:      # a broken/unfinished component must not crash the others
        print(f"[WARN] component {path} not loaded: {e}")
```

Each component's `router.py` exports `router = APIRouter(tags=["C1 – Document Understanding"])`.

**Standalone dev runner** (`dev_app.py`) so a member can run only their component:

```python
from fastapi import FastAPI
from .router import router
app = FastAPI(title="C2 dev")
app.include_router(router, prefix="/api/v1/c2")
# run: uvicorn app.components.c2_case_analysis.dev_app:app --reload --port 8002
```

Suggested dev ports: C1 → 8001, C2 → 8002, C3 → 8003, C4 → 8004, full app → 8000.

---

## 9. API Conventions (all components)

- Base path: `/api/v1/c{n}/...`
- JSON in / JSON out; Pydantic schemas for every request and response.
- Async endpoints (`async def`) and async DB sessions.
- Long tasks: return `202 Accepted` + resource id; client polls `GET /.../{id}` (status: `queued | processing | done | failed`).
- Streaming (C3): Server-Sent Events.
- Uniform error body:
  ```json
  { "error": { "code": "DOCUMENT_NOT_FOUND", "message": "…", "details": {} } }
  ```
- Pagination: `?limit=20&offset=0`.
- Auth: JWT bearer token dependency from `app.core.security`; every user-owned resource stores `user_id`.
- Every AI response that states legal content includes `sources` and a `disclaimer`.
- Version stamp on AI outputs: `model_version`, `prompt_version`, `scoring_version` (C2).
- CORS: allow the Next.js origin from `CORS_ORIGINS`.

---

## 10. LangChain / LangGraph Guidelines

- Get models only via `app.core.llm.get_llm()` and `app.core.embeddings.get_embedder()`.
- Use **LCEL** chains for simple flows (C3 RAG) and **LangGraph** for multi-step agents (C2 pipeline, C4 verifier). Keep graph state as a typed `TypedDict`/Pydantic model.
- Use `llm.with_structured_output(PydanticModel)` for anything the API returns as JSON.
- Prompts live in each component's `prompts/` folder as versioned files (e.g., `explain_v1.md` or Python templates) — not inline strings scattered in code.
- Grounding rule in every legal prompt: *"Use only the provided sources. If they are insufficient, say so. Cite every claim."*
- Temperature 0 for extraction/verification; low for answers.
- Add retries + timeouts around LLM calls; log token usage.
- Observability: enable LangSmith tracing (optional) with one project per component.
- Never put secrets or full user documents in logs.

---

## 11. Data Flow Between Components (contract-based)

```
C1 ──StructuredLegalInfo──► C2   (case analysis input)
C1 ──StructuredLegalInfo──► C3   (context for questions about an uploaded document)
C2 ──AnalysisResult───────► C3   (answer questions about the analysis)   [via C2 public.py]
C3 ──answer text──────────► C4   (verify generated answers)              [via C4 public.py]
C1/C2/C3/C4 ──────────────► shared.retrieval ──► shared.legal_chunks (read-only)
```

Integration happens **through `public.py` functions + `shared/contracts`**, so any component can be swapped with a mock during development.

---

## 12. Team Workflow

**Ownership (`CODEOWNERS`)**

```
/app/components/c1_document_understanding/   @member1
/app/components/c2_case_analysis/            @member2
/app/components/c3_legal_qa/                 @member3
/app/components/c4_misinformation/           @member4
/app/core/                                   @lead @member1 @member2 @member3 @member4
/app/shared/                                 @lead @member1 @member2 @member3 @member4
/scripts/                                    @lead
```

**Git**
- `main` (protected) ← `develop` ← feature branches: `c2/relevance-ranker`, `c3/sse-streaming`, etc.
- Branch prefix = component id. A PR touching more than one component folder (other than `core`/`shared`) is rejected.
- Conventional commits: `feat(c2): add evidence gap analysis`.
- CI: lint (`ruff`), type-check (`mypy`, optional), tests per component (`pytest app/components/c2_case_analysis`).

**Suggested build order**
1. **Week 0 (whole team):** finalize `core/`, `shared/contracts`, `shared/retrieval` skeleton, Neon setup + branches, ingest a small sample corpus.
2. **Phase 1:** each member builds their component against mocks / sample corpus using `dev_app.py`.
3. **Phase 2:** connect via `public.py`, integration tests, frontend wiring.
4. **Phase 3:** evaluation, tuning (C2 weights, C4 thresholds), hardening.

---

## 13. Environment Variables (`.env.example`)

```
APP_ENV=development
DATABASE_URL=postgresql+asyncpg://USER:PASS@HOST/DB?ssl=require        # Neon pooled
DATABASE_URL_DIRECT=postgresql://USER:PASS@HOST/DB?sslmode=require     # Neon direct (Alembic)
JWT_SECRET=change-me
JWT_EXPIRE_MINUTES=60
CORS_ORIGINS=http://localhost:3000

LLM_PROVIDER=anthropic          # or openai, etc.
LLM_MODEL=<model-name>
LLM_API_KEY=
EMBEDDING_MODEL=<model-name>
EMBEDDING_DIM=768

FILE_STORAGE=local              # local | s3
FILE_STORAGE_PATH=./data/uploads
LANGSMITH_TRACING=false
```

Each component may add its own vars prefixed `C1_`, `C2_`, … in its `config.py`; add them to `.env.example` in the same PR.

---

## 14. Cursor Rules (put in `.cursorrules` or `.cursor/rules/backend.mdc`)

```
You are working on the Smart Civil Case Analysis Platform backend.
Read BACKEND_ARCHITECTURE.md before writing code.

- I am working on component: <C1 | C2 | C3 | C4>. Only create/edit files inside
  app/components/<my_component_folder>/ (plus my own tests).
- Never modify other components, app/core, or app/shared unless I explicitly say so.
- Never import another component's internals. Use app.shared.* or their public.py only.
- Use async FastAPI endpoints, Pydantic v2 schemas, SQLAlchemy 2.x async, schema "<cN>" for all tables.
- Get LLM/embeddings only from app.core.llm / app.core.embeddings.
- Retrieval over the legal corpus must go through app.shared.retrieval.HybridRetriever.
- Follow the folder structure, endpoint list, and table names in the architecture file.
- All AI outputs must be grounded in retrieved sources and include citations.
- Add tests in the component's tests/ folder for every service function.
```

**Example prompts for Cursor**

- *"Read BACKEND_ARCHITECTURE.md. I'm Member 2 (C2). Implement `pipeline/ranker.py` with the relevance score formula from §6.2, weights loaded from `config.py`, plus unit tests."*
- *"Read BACKEND_ARCHITECTURE.md. I'm Member 1 (C1). Implement `pipeline/pdf_extractor.py` with PyMuPDF and the `POST /documents` endpoint from §6.1."*
- *"Read BACKEND_ARCHITECTURE.md. I'm Member 3 (C3). Build the RAG chain in `rag/answer_chain.py` using HybridRetriever and return `sources`."*
- *"Read BACKEND_ARCHITECTURE.md. I'm Member 4 (C4). Implement `pipeline/verifier.py` with the three verdicts and the default-to-Insufficient rule."*

---

## 15. Testing & Quality

- **Unit tests** per component (`tests/`), mocking LLM and retrieval (`FakeListChatModel`, fixtures).
- **Contract tests**: validate `public.py` return types against `shared/contracts`.
- **Integration tests** in `/tests/integration` (added in Phase 2).
- **Evaluation notebooks/scripts** live in each component's `evaluation/` (C2 retrieval metrics, C4 verdict accuracy, C1 classification/NER F1).
- Pre-commit: `ruff`, `ruff format`, `pytest -q`.

---

## 16. Security & Legal Safeguards

- Uploaded documents are sensitive: per-user access checks on every read/delete, restrict file types and size, virus-scan if possible, delete on request.
- Do not log document contents or personal data.
- Rate-limit LLM endpoints (e.g., `slowapi`).
- Prompt-injection defence: treat uploaded text and retrieved chunks as **data, not instructions**; keep system prompts separate.
- All legal outputs carry the disclaimer: *"This is informational and not legal advice."*
- Explicitly separate **actual-case analysis** from **hypothetical (what-if)** outputs.

---

## 17. Open Decisions (settle in Week 0)

1. Embedding model for retrieval (raw Legal-BERT is not trained for sentence similarity; consider a sentence-embedding model or a fine-tuned Legal-BERT) and its vector dimension.
2. LLM provider/model and budget.
3. Language scope: English only, or Sinhala/Tamil support (affects OCR, tokenization, embeddings).
4. Corpus sources and licensing (Acts, Court of Appeal / Supreme Court judgments) and ingestion schedule.
5. Auth approach shared with the Next.js frontend (own JWT vs. NextAuth-issued tokens).
6. File storage for uploads (local / S3 / R2).
7. Whether a task queue is needed (start without one).
