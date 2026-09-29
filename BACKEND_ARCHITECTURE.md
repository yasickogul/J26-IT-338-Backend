# Smart Civil Case Analysis Platform — Backend Architecture Guide

> **For Cursor / AI assistants:** This file is the source of truth for the backend. Before generating any code, read this file fully. Follow the folder structure, ownership rules, and contracts exactly. Never edit files outside the component you were asked to work on.

---

## 1. Project Summary

An AI-powered legal assistance platform for **Sri Lankan civil law**. Users submit a legal problem or upload legal documents; the system understands the input, retrieves relevant Sri Lankan legal sources and precedents, generates case intelligence reports, verifies legal claims, simulates adversarial legal arguments, and returns **structured, explainable, source-cited** results.

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

**The platform is NOT one linear pipeline.** It is four independent components, each with its own input, its own output, and its own API. A user (or the frontend) can call any one of them directly. Some components *may optionally* use another component's output as extra context (e.g. C3 can be given a document already processed by C1), but none of them require it — each must work standalone on its own input.

### The four components

| ID | Component | Owner | Input | Output |
|---|---|---|---|---|
| C1 | Legal Document Classification & Information Extraction | Member 1 | An uploaded legal document (PDF/text) or pasted text | `StructuredLegalInfo` — document type, entities, case type, claims, parties, evidence, court |
| C2 | Legal Document Intelligence & Civil Case Insight Generation | Member 2 | A civil case document (uploaded PDF or pasted text) | A **Civil Case Intelligence Report**: case information, parties/positions, legal issues, evidence/information gaps, similar cases, relevant laws/precedents, explanations and citations |
| C3 | AI-Powered Legal Misinformation Detection System | Member 3 | A legal claim as **text or an image** (OCR), in **English, Sinhala, or Tamil** | Per-claim verdict — **True / False / Misleading** — with explanation and cited sources |
| C4 | Multi-Agent Legal Argumentation Engine | Member 4 | Case facts (structured input) | A structured **argument graph** (JSON nodes/edges) from a simulated Plaintiff-vs-Defense debate, audited for factual grounding |

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
                      └──┬─────────┬─────────┬───────────────┬┘
          /api/v1/c1     │  /c2    │  /c3    │  /c_argumentation
          ┌──────────────▼┐ ┌──────▼──────┐ ┌▼─────────────┐ ┌▼──────────────┐
          │ C1 Case       │ │ C2 Case     │ │ C3 Misinfo   │ │ C4 Argumen-   │
          │ Analysis      │ │ Analysis    │ │ Detection    │ │ tation Engine │
          └──────┬────────┘ └──────┬──────┘ └──────┬───────┘ └─────┬─────────┘
                 │                 │               │               │
        ┌────────▼─────────────────▼───────────────▼───────────────▼───────┐
        │  app/core  (config, db, llm, embeddings)  +  app/shared          │
        │  (contracts, retrieval service, citation utils)                  │
        └────────────────────────────┬─────────────────────────────────────┘
                                     │
                     ┌───────────────▼──────────────────┐
                     │ Neon PostgreSQL + pgvector        │
                     │ schemas: shared · c1 · c2 · c3 · c_arg │
                     └────────────────────────────────────┘
```

Each box above is independent: it accepts its own request and returns its own response. None of them sit *inside* another's request path — the arrows into `app/shared` are the only thing they have in common (the legal corpus and retrieval service), and that access is read-only.

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
│       ├── c1_CaseAnalysis/     # 👤 Member 1 ONLY
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
│       ├── c2_case_intelligence/          # 👤 Member 2 ONLY
│       │   ├── __init__.py
│       │   ├── public.py
│       │   ├── router.py                  # → /api/v1/c2
│       │   ├── dev_app.py
│       │   ├── config.py                  # scoring weights, top_k, thresholds
│       │   ├── schemas.py                 # incl. CivilCaseIntelligenceReport
│       │   ├── models.py                  # schema "c2" (documents, reports, report_cases, ...)
│       │   ├── repository.py
│       │   ├── service.py
│       │   ├── pipeline/
│       │   │   ├── document_processor.py        # PDF/text extraction (PyMuPDF), cleaning, segmentation
│       │   │   ├── info_extractor.py            # legal information extraction (case info, parties, issues, evidence, remedy)
│       │   │   ├── insight_analyzer.py          # civil case insight analysis (issues, positions, information gaps)
│       │   │   ├── query_builder.py             # query representation for retrieval
│       │   │   ├── similar_case_retriever.py    # similar historical case retrieval (uses shared.retrieval)
│       │   │   ├── ranker.py                    # relevance percentage scoring of similar cases
│       │   │   ├── comparator.py                # fact & evidence comparison vs. similar cases
│       │   │   ├── evidence_gap.py              # evidence/information gaps
│       │   │   ├── law_precedent_retriever.py   # RAG over laws & precedents (uses shared.retrieval)
│       │   │   └── report_builder.py            # assembles the Civil Case Intelligence Report
│       │   ├── agents/
│       │   │   ├── explanation_agent.py   # LangChain: grounded explanations with citations
│       │   │   └── graph.py               # LangGraph workflow
│       │   ├── prompts/
│       │   ├── evaluation/                # extraction accuracy, precision/recall/F1, retrieval metrics
│       │   ├── migrations/versions/       # branch "c2"
│       │   ├── tests/
│       │   └── README.md
│       │
│       ├── c3_misinformation/             # 👤 Member 3 ONLY
│       │   ├── __init__.py
│       │   ├── public.py
│       │   ├── router.py                  # → /api/v1/c3
│       │   ├── dev_app.py
│       │   ├── config.py
│       │   ├── schemas.py
│       │   ├── models.py                  # schema "c3" (claims, verifications, media_inputs)
│       │   ├── repository.py
│       │   ├── service.py
│       │   ├── pipeline/
│       │   │   ├── input_router.py        # routes text vs. image input
│       │   │   ├── ocr_extractor.py       # image → text OCR (English/Sinhala/Tamil)
│       │   │   ├── language_detector.py   # detects en / si / ta
│       │   │   ├── translator.py          # normalizes si/ta text for retrieval, if needed
│       │   │   ├── claim_extractor.py
│       │   │   ├── concept_identifier.py
│       │   │   ├── source_retriever.py    # uses shared.retrieval
│       │   │   ├── comparator.py          # semantic / NLI evidence comparison
│       │   │   └── verifier.py            # True / False / Misleading
│       │   ├── agents/
│       │   │   └── verification_agent.py
│       │   ├── prompts/
│       │   ├── evaluation/
│       │   ├── migrations/versions/       # branch "c3"
│       │   ├── tests/
│       │   └── README.md
│       │
│       └── c_argumentation/               # 👤 Member 4 ONLY
│           ├── __init__.py
│           ├── public.py
│           ├── router.py                  # → /api/v1/c_argumentation
│           ├── dev_app.py
│           ├── config.py
│           ├── schemas.py
│           ├── models.py                  # schema "c_arg" (cases, argument_nodes, argument_edges)
│           ├── repository.py
│           ├── service.py
│           ├── pipeline/
│           │   ├── case_intake.py             # parses submitted case facts
│           │   ├── source_retriever.py        # uses shared.retrieval (see note in §6.4)
│           │   ├── debate_orchestrator.py     # LangGraph turn-taking between Plaintiff/Defense
│           │   ├── argument_graph_builder.py  # builds JSON node/edge graph from turns
│           │   └── auditor_pipeline.py        # claim-vs-source check, strength scoring
│           ├── agents/
│           │   ├── plaintiff_agent.py
│           │   ├── defense_agent.py
│           │   └── auditor_agent.py
│           ├── prompts/
│           ├── evaluation/
│           ├── migrations/versions/       # branch "c_arg"
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

1. **Own your folder only.** Each member edits only their own folder (`app/components/c1_CaseAnalysis/`, `c2_case_intelligence/`, `c3_misinformation/`, or `c_argumentation/`). Never edit another component's folder.
2. **No cross-imports of internals.** A component must NOT do `from app.components.c2_case_intelligence.service import ...`. Allowed imports:
   - `app.core.*`
   - `app.shared.*`
   - Another component's `public.py` (a thin facade; only functions/types listed there).
3. **Cross-component data goes through contracts** in `app/shared/contracts/`. Example: C1's output is `StructuredLegalInfo`; any other component that *chooses* to use a C1 document as context consumes that type, not C1's internal models. Using another component's output is always optional, never required.
4. **Changes to `core/` or `shared/`** need a PR reviewed by all 4 members (or the team lead). If you need something new, open a PR that adds it — don't fork a private copy.
5. **Database ownership:** each component has its own Postgres schema (`c1`, `c2`, `c3`, `c_arg`) and only writes to its own. The `shared` schema (legal corpus) is **read-only** for components; only `scripts/` ingestion writes to it.
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
    """Output of C1. Optional context input for C2 / C4 (and C3 when checking a claim
    that references an already-uploaded document)."""
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

Used by C2, C3, and (optionally, see §6.4) C4, so nobody re-implements retrieval.

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
→ StructuredLegalInfo → saved in c1 schema → returned to the caller
(other components may optionally fetch it via `public.py` if a request references a `document_id`)
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

### 6.2 C2 — Legal Document Intelligence & Civil Case Insight Generation

**Purpose.** Takes a civil case document and produces one **Civil Case Intelligence Report**: what the case is about, who the parties are and what position each takes, the legal issues, what evidence or information is missing, which similar historical cases exist, which laws and precedents apply, and grounded explanations with citations.

**Pipeline**

```
Civil Case Document → Document Processing → Legal Information Extraction
→ Civil Case Insight Analysis → Similar Historical Case Retrieval
→ RAG for Laws & Precedents → Grounded Explanation → Civil Case Intelligence Report
```

| Step | Module(s) in `pipeline/` or `agents/` | What it does |
|---|---|---|
| Document Processing | `document_processor.py` | PyMuPDF text extraction (OCR fallback for scanned PDFs), cleaning, sentence/section segmentation, language detection |
| Legal Information Extraction | `info_extractor.py` | Extracts case type, court, case number, parties, claims, remedy requested, evidence mentioned, dates, legal references (NER + rules + LLM with structured output) |
| Civil Case Insight Analysis | `insight_analyzer.py` | Identifies the legal issues, each party's position, and information gaps inside the document |
| Similar Historical Case Retrieval | `query_builder.py`, `similar_case_retriever.py`, `ranker.py`, `comparator.py`, `evidence_gap.py` | Hybrid retrieval (BM25 + pgvector) of similar judgments, percentage-based relevance ranking, fact/evidence comparison, evidence gaps compared with similar cases |
| RAG for Laws & Precedents | `law_precedent_retriever.py` | Retrieves relevant Acts/sections and precedent passages from the shared legal corpus for the identified legal issues |
| Grounded Explanation | `agents/explanation_agent.py` | Explains, using only retrieved sources, why each similar case and law/precedent is relevant; every statement carries a citation |
| Report | `report_builder.py` | Assembles the final Civil Case Intelligence Report |

LangGraph flow (`agents/graph.py`): `process_document → extract_info → analyse_insights → retrieve_similar_cases → retrieve_laws_precedents → explain → build_report` (each a node; state is a typed dict).

**Endpoints (`/api/v1/c2`)**

| Method | Path | Description |
|---|---|---|
| POST | `/reports` | Submit a civil case document (multipart file, or JSON `{ "text": "..." }`) → `report_id` (`202`, runs asynchronously) |
| GET | `/reports/{id}` | Status + the full Civil Case Intelligence Report once done |
| GET | `/reports/{id}/similar-cases` | Ranked similar cases with percentage relevance score + factor breakdown |
| GET | `/reports/{id}/similar-cases/{case_id}/explanation` | Similarities, differences, reasoning |
| GET | `/reports/{id}/laws-precedents` | Relevant laws and precedents with citations |
| GET | `/reports/{id}/gaps` | Evidence/information gaps |
| GET | `/reports` | User's history |
| DELETE | `/reports/{id}` | Delete the report and the uploaded document |

**Main output — Civil Case Intelligence Report** (Pydantic model `CivilCaseIntelligenceReport` in this component's `schemas.py`; it is not shared with other components)

```json
{
  "report_id": "…",
  "status": "done",
  "case_information": {
    "case_type": "Employment",
    "court": "Court of Appeal",
    "case_number": "…",
    "summary": "…",
    "remedy_requested": "Compensation"
  },
  "parties_and_positions": [
    { "party": "…", "role": "plaintiff | defendant | appellant | respondent | other", "position": "…" }
  ],
  "legal_issues": [
    { "issue": "Wrongful termination", "description": "…" }
  ],
  "evidence_and_information_gaps": [
    { "kind": "evidence | information", "description": "…", "basis": "…" }
  ],
  "similar_cases": [
    {
      "source_id": "…", "title": "…", "citation": "…",
      "relevance_score": 91,
      "factor_labels": { "issue": "High", "fact": "High", "claim": "High", "evidence": "Medium", "remedy": "High" },
      "explanation": { "similarities": ["…"], "differences": ["…"], "summary": "…" }
    }
  ],
  "relevant_laws_and_precedents": [
    { "citation": { "…": "SourceCitation" }, "excerpt": "…", "why_relevant": "…" }
  ],
  "explanations": { "overall_summary": "…" },
  "citations": [ { "…": "SourceCitation" } ],
  "scoring_version": "…",
  "model_version": "…",
  "disclaimer": "This is informational and not legal advice."
}
```

**Report rules**
- Every entry in `similar_cases` and `relevant_laws_and_precedents` carries a citation; the top-level `citations` list is the de-duplicated union of them.
- Gaps are phrased as "not found in the document" or "present in similar cases but absent here", never as legal conclusions.
- If retrieval finds no strong similar cases or laws, return the sections empty and say so, rather than padding them with weak matches.

**Similar-case relevance scoring (percentage-based relevance score — NOT a probability)**

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

**Grounded explanation (LangChain agent / chain)**
- Input: extracted case information + insights + retrieved similar cases and law/precedent chunks + factor scores.
- Output (structured, Pydantic): per similar case `similarities[]`, `differences[]`, `factor_reasons{}`, `summary`; per law/precedent `why_relevant`; plus an overall summary.
- The LLM must only use provided evidence; every statement links to a chunk/citation. Use `with_structured_output(...)`.

**Notes**
- Long-running work → FastAPI `BackgroundTasks` first; move to a task queue only if needed. Status field: `queued → processing → done | failed`.
- Uploaded files go to object storage (S3/R2) or local disk in dev; save only the path in the DB.
- C2 does its own document processing and extraction inside its folder; it does not import from C1.

**Tables (schema `c2`)**: `documents` (id, user_id, filename, mime, storage_path, status, created_at), `reports` (id, document_id, user_id, status, report_json JSONB, scoring_version, model_version, created_at), `report_cases` (report_id, case_id, score, factor_scores JSONB, scoring_version), `explanations` (report_id, case_id, explanation_json JSONB), `evidence_gaps` (report_id, kind, description, basis).

**Evaluation (`evaluation/`)**: field-level extraction accuracy/F1 against labelled documents; Precision@k, Recall@k, F1, MRR/nDCG and confusion matrix on a labelled relevance set; a groundedness check that every citation in a report resolves to a retrieved chunk.

**`public.py` exposes:** `async def generate_report(text: str) -> CivilCaseIntelligenceReport`.

---

### 6.3 C3 — AI-Powered Legal Misinformation Detection System

**Input is text OR an image (OCR), in English, Sinhala, or Tamil.** This is the component's main technical challenge: everything downstream (claim extraction, retrieval, verdict) has to work regardless of which of the two input modes and three languages arrived.

**Pipeline**

```
Input (text OR image) → input routing → [image path: OCR extraction (en/si/ta)]
→ language detection → [if si/ta: translate/normalize for retrieval] → claim extraction
(atomic claims) → legal concept identification → relevant source retrieval
→ semantic/evidence comparison (NLI or LLM judge) → verdict per claim:
True | False | Misleading → explanation + sources (in the input language)
```

**Endpoints (`/api/v1/c3`)**

| Method | Path | Description |
|---|---|---|
| POST | `/verifications` | Submit a text claim (JSON: `{ "text": "...", "language": "auto\|en\|si\|ta" }`) → `verification_id` |
| POST | `/verifications/image` | Submit an image (multipart) → runs OCR, then verification |
| GET | `/verifications/{id}` | Verdict(s) per atomic claim, explanation, evidence, `detected_language`, `input_mode` |
| POST | `/verify-answer` | Verify arbitrary generated text (e.g. from another component) against sources |
| GET | `/verifications` | User's history |

**Response shape**

```json
{
  "input_mode": "text | image",
  "detected_language": "en | si | ta",
  "claims": [
    {
      "claim": "…",
      "claim_translated": "… (English, if original was si/ta)",
      "verdict": "True | False | Misleading",
      "confidence": 0.82,
      "explanation": "…",
      "evidence": [ { "citation": {…}, "excerpt": "…", "stance": "supports|contradicts" } ]
    }
  ]
}
```

**Rules**
- Only three verdicts exist: **True**, **False**, **Misleading** (no "insufficient evidence" bucket). When retrieved evidence is weak or ambiguous, still return the best-supported verdict but keep `confidence` low and say so explicitly in `explanation` — never silently present a low-confidence guess as certain. A claim that is technically accurate but omits crucial context is **Misleading**, not True.
- Combine an NLI/cross-encoder model (entailment / contradiction / neutral) with an LLM judge for the explanation.
- OCR: PyMuPDF/pdf text extraction won't help here since input is an image — use Tesseract with `eng+sin+tam` trained data (or a cloud OCR API that supports Sinhala/Tamil scripts) in `pipeline/ocr_extractor.py`.
- Language detection needs Sinhala/Tamil support specifically — common libraries like `langdetect` are unreliable for these scripts; prefer something like fastText's `lid.176` model, which covers `si` and `ta`.
- The shared legal corpus (`shared.legal_chunks`) is expected to be mostly English (Acts, judgments). If the claim is in Sinhala/Tamil, translate/normalize the extracted claim into English for retrieval (`pipeline/translator.py`), but return the explanation in the original input language when possible.
- Evaluate with a labelled claim set: accuracy and per-class (True/False/Misleading) precision/recall/confusion matrix, plus OCR accuracy and language-detection accuracy as separate metrics.

**Tables (schema `c3`)**: `verifications` (id, input_mode, detected_language, status), `claims` (verification_id, claim_text, claim_translated, verdict, confidence), `claim_evidence`, `media_inputs` (verification_id, storage_path, ocr_raw_text, ocr_confidence).

**`public.py` exposes:** `async def verify_claim(text: str, language: str | None = None) -> VerificationResult`, `async def verify_image(image_bytes: bytes) -> VerificationResult`.

---

### 6.4 C4 — Multi-Agent Legal Argumentation Engine

**Purpose.** Rather than a single verdict or a ranked list, this component simulates an **adversarial legal debate**: a Plaintiff agent and a Defense agent argue opposing sides of the submitted case over several rounds, each citing retrieved sources, while an Auditor agent checks every claim against the actual source text and scores how well-grounded it is. The output is a structured, inspectable **argument graph**, not prose.

**Pipeline**

```
Case facts → case_intake (parse facts) → source_retriever (retrieval for both sides)
→ debate_orchestrator: LangGraph turn-taking loop
      Plaintiff turn → Defense turn → Auditor check → (repeat for N rounds)
→ argument_graph_builder (turns → JSON nodes/edges)
→ auditor_pipeline (final claim-vs-source check, strength scoring per node)
→ Argument graph + audit report
```

**Endpoints (`/api/v1/c_argumentation`)** — note this component's prefix is `c_argumentation`, not `c4`, because its folder isn't named `c4_*`.

| Method | Path | Description |
|---|---|---|
| POST | `/debates` | Submit case facts (structured input or free text) → `debate_id`, runs the debate |
| GET | `/debates/{id}` | Status + full argument graph (nodes + edges) |
| GET | `/debates/{id}/turns` | Ordered transcript: each Plaintiff/Defense turn in sequence |
| GET | `/debates/{id}/audit` | Auditor's per-node claim-vs-source findings + strength scores |
| POST | `/debates/{id}/continue` | Run additional round(s) on an existing debate |
| GET | `/debates` | User's history |

**Argument graph shape**

```json
{
  "nodes": [
    { "id": "n1", "agent": "plaintiff", "round": 1, "claim": "…", "cites": ["src_12"], "strength_score": 0.74 },
    { "id": "n2", "agent": "defense",   "round": 1, "claim": "…", "cites": ["src_7"],  "strength_score": 0.61 }
  ],
  "edges": [
    { "from": "n2", "to": "n1", "relation": "attacks" },
    { "from": "n1", "to": "src_12", "relation": "cites" }
  ]
}
```

- `strength_score` comes from the Auditor: how well the node's claim is actually supported by the cited source text (not just whether a citation is present).
- `relation` values: `attacks`, `supports`, `cites`, `rebuts`.

**Agents (`agents/`)**
- **`plaintiff_agent.py`** — argues for the claimant, retrieving and citing precedents/sources that support their position.
- **`defense_agent.py`** — argues for the respondent; retrieves counter-precedents and attacks weak points in the Plaintiff's claims.
- **`auditor_agent.py`** — after each round (or at the end), checks every claim against the *actual retrieved source text*, not the agent's paraphrase, flags unsupported assertions, and assigns `strength_score`.

**`debate_orchestrator.py` (LangGraph)**: a graph with nodes `plaintiff_turn → defense_turn → auditor_check → (loop for N rounds or until a stopping condition) → finalize_graph`. State is a typed dict: `case_facts`, `turn_history`, `nodes`, `edges`, `round_count`.

**Retrieval note:** the component description names **ChromaDB** for this component's RAG. To keep contracts consistent across the platform, either (a) reuse `app.shared.retrieval.HybridRetriever` like C2/C3 do (simplest, one less moving part), or (b) run a separate ChromaDB collection here but have `source_retriever.py` wrap it so it still returns `list[RetrievedChunk]` — the same contract type the rest of the platform uses. Decide this in Week 0 (see §17) since it affects whether this component needs its own vector store alongside Neon/pgvector.

**Tables (schema `c_arg`)**: `cases` (id, facts_json, user_id, created_at), `argument_nodes` (id, case_id, agent, round, claim_text, cites JSONB, strength_score, created_at), `argument_edges` (id, case_id, from_node, to_node, relation).

**`public.py` exposes:** `async def run_debate(case_facts: dict) -> ArgumentGraph`.

---

## 7. Database & Migrations (Neon)

- **One Neon database**, five schemas: `shared`, `c1`, `c2`, `c3`, `c_arg`. Each component's SQLAlchemy models set `__table_args__ = {"schema": "..."}` (`"c1"`, `"c2"`, `"c3"`, or `"c_arg"`) and use their **own** `DeclarativeBase`/`MetaData`.
- **Alembic multi-branch:** in `alembic.ini` list every `version_locations`; each migration file declares `branch_labels = ("c1",)` (etc., using `"c_arg"` for the argumentation engine). Members run `alembic upgrade c1@head` for their own branch only.
  ```
  version_locations = app/components/c1_CaseAnalysis/migrations/versions
                      app/components/c2_case_intelligence/migrations/versions
                      app/components/c3_misinformation/migrations/versions
                      app/components/c_argumentation/migrations/versions
                      app/shared/migrations/versions
  ```
- **Dev isolation on Neon:** use **Neon branches** — every member gets their own database branch (copy-on-write) so experiments never break each other's data. `main` branch = integration/staging.
- Cross-component foreign keys are **not allowed**. Reference other components by plain UUID columns (no FK) and resolve through `public.py`.

---

## 8. Component Registration & App Bootstrap

Because `c_argumentation`'s folder name doesn't follow the `cN_*` pattern, prefixes are declared **explicitly** as a dict rather than derived from the folder name:

```python
# app/registry.py  — module path → URL prefix segment
COMPONENTS = {
    "app.components.c1_CaseAnalysis": "c1",
    "app.components.c2_case_intelligence": "c2",
    "app.components.c3_misinformation": "c3",
    "app.components.c_argumentation": "c_argumentation",
}
```

```python
# app/main.py
from importlib import import_module
from fastapi import FastAPI
from app.registry import COMPONENTS

app = FastAPI(title="Smart Civil Case Analysis Platform", version="0.1.0")

for module_path, prefix in COMPONENTS.items():
    try:
        module = import_module(f"{module_path}.router")
        app.include_router(module.router, prefix=f"/api/v1/{prefix}")
    except ImportError as e:      # a broken/unfinished component must not crash the others
        print(f"[WARN] component {module_path} not loaded: {e}")
```

Each component's `router.py` exports `router = APIRouter(tags=["C1 – Case Analysis"])`.

**Standalone dev runner** (`dev_app.py`) so a member can run only their component:

```python
from fastapi import FastAPI
from .router import router
app = FastAPI(title="C2 dev")
app.include_router(router, prefix="/api/v1/c2")
# run: uvicorn app.components.c2_case_intelligence.dev_app:app --reload --port 8002
```

Suggested dev ports: C1 → 8001, C2 → 8002, C3 (misinformation) → 8003, C4 / `c_argumentation` → 8004, full app → 8000.

---

## 9. API Conventions (all components)

- Base path: `/api/v1/c{n}/...`
- JSON in / JSON out; Pydantic schemas for every request and response.
- Async endpoints (`async def`) and async DB sessions.
- Long tasks: return `202 Accepted` + resource id; client polls `GET /.../{id}` (status: `queued | processing | done | failed`).
- Streaming: use Server-Sent Events for any long, incremental output (e.g. `c_argumentation` streaming debate turns as they're generated).
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
- Use **LCEL** chains for simple flows (single-shot extraction/verification) and **LangGraph** for multi-step agents (C2's document-to-report pipeline, C3's OCR→verify pipeline, and especially `c_argumentation`'s Plaintiff/Defense/Auditor turn-taking loop). Keep graph state as a typed `TypedDict`/Pydantic model.
- Use `llm.with_structured_output(PydanticModel)` for anything the API returns as JSON.
- Prompts live in each component's `prompts/` folder as versioned files (e.g., `explain_v1.md` or Python templates) — not inline strings scattered in code.
- Grounding rule in every legal prompt: *"Use only the provided sources. If they are insufficient, say so. Cite every claim."*
- Temperature 0 for extraction/verification; low for answers.
- Add retries + timeouts around LLM calls; log token usage.
- Observability: enable LangSmith tracing (optional) with one project per component.
- Never put secrets or full user documents in logs.

---

## 11. Data Flow Between Components — Independent, Not a Pipeline

Each component has its **own input and its own output**, and a user can call any one directly without touching the others — none of them requires another to already have run. The only thing every component shares is read-only access to the legal corpus:

```
                         ┌────────────────────────────────────────┐
                         │  shared.retrieval → shared.legal_chunks │
                         │              (read-only)                │
                         └───▲──────────▲──────────▲───────────────┘
                             │          │          │
   C1                          C2                              C3                       C4
   document                    civil case document             claim (text/image)       case facts
     → StructuredLegalInfo       → Civil Case Intelligence       → True/False/Misleading  → argument graph
                                   Report                          verdicts
```

C2 does its own document processing and legal information extraction inside its own folder (`pipeline/document_processor.py`, `pipeline/info_extractor.py`), so it needs nothing from C1.

The **only** optional cross-component link is:

```
C1 ──StructuredLegalInfo (optional)──► C4   (a debate can be seeded from an already-uploaded document)
```

It is opt-in: the caller passes a `document_id` if they have one; the endpoint also accepts plain structured/free-text input so it works with no other component involved. There is no required chain C1→C2→C3→C4 — build and test each component against its own input/output contract independently.

Where a link exists, integration happens **through `public.py` functions + `shared/contracts`**, so any component can be swapped with a mock during development.

---

## 12. Team Workflow

**Ownership (`CODEOWNERS`)**

```
/app/components/c1_CaseAnalysis/   @member1
/app/components/c2_case_intelligence/          @member2
/app/components/c3_misinformation/           @member3
/app/components/c_argumentation/             @member4
/app/core/                                   @lead @member1 @member2 @member3 @member4
/app/shared/                                 @lead @member1 @member2 @member3 @member4
/scripts/                                    @lead
```

**Git**
- `main` (protected) ← `develop` ← feature branches: `c2/relevance-ranker`, `c3/sse-streaming`, etc.
- Branch prefix = component id. A PR touching more than one component folder (other than `core`/`shared`) is rejected.
- Conventional commits: `feat(c2): add evidence gap analysis`.
- CI: lint (`ruff`), type-check (`mypy`, optional), tests per component (`pytest app/components/c2_case_intelligence`).

**Suggested build order**
1. **Week 0 (whole team):** finalize `core/`, `shared/contracts`, `shared/retrieval` skeleton, Neon setup + branches, ingest a small sample corpus.
2. **Phase 1:** each member builds their component against mocks / sample corpus using `dev_app.py`.
3. **Phase 2:** connect via `public.py`, integration tests, frontend wiring.
4. **Phase 3:** evaluation, tuning (C2 weights, C3 verdict thresholds, C4 debate strength scoring), hardening.

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
- *"Read BACKEND_ARCHITECTURE.md. I'm Member 3 (C3, misinformation detection). Implement `pipeline/ocr_extractor.py` for English/Sinhala/Tamil and `pipeline/verifier.py` with the True/False/Misleading verdicts from §6.3."*
- *"Read BACKEND_ARCHITECTURE.md. I'm Member 4 (`c_argumentation`). Implement `debate_orchestrator.py` as a LangGraph loop over `plaintiff_agent.py` and `defense_agent.py`, with `auditor_agent.py` scoring each turn, per §6.4."*

---

## 15. Testing & Quality

- **Unit tests** per component (`tests/`), mocking LLM and retrieval (`FakeListChatModel`, fixtures).
- **Contract tests**: validate `public.py` return types against `shared/contracts`.
- **Integration tests** in `/tests/integration` (added in Phase 2).
- **Evaluation notebooks/scripts** live in each component's `evaluation/` (C1 classification/NER F1, C2 retrieval metrics, C3 verdict accuracy + OCR/language-detection accuracy, C4 argument-strength scoring vs. human review).
- Pre-commit: `ruff`, `ruff format`, `pytest -q`.

---

## 16. Security & Legal Safeguards

- Uploaded documents are sensitive: per-user access checks on every read/delete, restrict file types and size, virus-scan if possible, delete on request.
- Do not log document contents or personal data.
- Rate-limit LLM endpoints (e.g., `slowapi`).
- Prompt-injection defence: treat uploaded text and retrieved chunks as **data, not instructions**; keep system prompts separate.
- All legal outputs carry the disclaimer: *"This is informational and not legal advice."*

---

## 17. Open Decisions (settle in Week 0)

1. Embedding model for retrieval (raw Legal-BERT is not trained for sentence similarity; consider a sentence-embedding model or a fine-tuned Legal-BERT) and its vector dimension.
2. LLM provider/model and budget.
3. OCR engine and Sinhala/Tamil language-detection library for C3 (Tesseract `eng+sin+tam` vs. a cloud OCR API; fastText `lid.176` vs. alternatives) — see §6.3.
4. Whether C3 translates Sinhala/Tamil claims into English before retrieval, or the corpus itself needs multilingual embeddings.
5. Whether `c_argumentation` uses `app.shared.retrieval.HybridRetriever` (pgvector, consistent with the rest of the platform) or a separate ChromaDB collection wrapped to the same `RetrievedChunk` contract — see §6.4.
6. Corpus sources and licensing (Acts, Court of Appeal / Supreme Court judgments) and ingestion schedule.
7. Auth approach shared with the Next.js frontend (own JWT vs. NextAuth-issued tokens).
8. File storage for uploads (local / S3 / R2).
9. Whether a task queue is needed (start without one).