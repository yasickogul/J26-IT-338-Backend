# C1 – Case Analysis

Owner: @member

Stage 2 (implemented): **Current case intake** and **legal information extraction**.  
Stage 3 (implemented in this component): **C1-owned semantic similar-judgment retrieval** + grounded analysis.

- Component path: `app/components/c1_CaseAnalysis/`
- API prefix: `/api/v1/c1`
- Standalone: `uvicorn app.components.c1_CaseAnalysis.dev_app:app --reload --port 8001`

## Stage 2 endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/intake` | JSON structured case intake → `ExtractedLegalInfo` |
| POST | `/intake/document` | Multipart PDF + `intake_json` form field |
| GET | `/intake/{id}` | Intake status + extraction result |
| POST | `/extract-text` | Free text → shared `StructuredLegalInfo` |
| GET | `/ping` | Health / stage marker |

## Stage 3 endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/similar-cases` | Free-text semantic search over `shared.legal_chunks` |
| POST | `/intake/{id}/similar-cases` | Retrieve + optional grounded analysis for a saved intake |
| GET | `/health/stage3` | Stage 3 readiness |

Semantic retrieval uses `app.core.embeddings.get_embedder()` (MiniLM 384-d) and pgvector cosine distance.  
It does **not** implement shared `HybridRetriever` (still a stub). Similarity scores are retrieval cosine similarity, not legal success probability.

Optional LM Studio enrichment: set `C1_LM_STUDIO_ENABLED=true` (and optional `C1_LM_STUDIO_MODEL`). Retrieval works without LM Studio.

Dispute categories (intake, not corpus): `contract`, `property_land`, `employment`, `rental`, `other_civil`.  
Corpus judgments remain `case_type="civil"`.
