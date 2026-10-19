# C1 – Case Analysis

Owner: @member

Stage 2 (implemented): **Current case intake** and **legal information extraction**.

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

Dispute categories (intake, not corpus): `contract`, `property_land`, `employment`, `rental`, `other_civil`.  
Corpus judgments remain `case_type="civil"`.

Stage 3 (similar-case retrieval / full intelligence report) is **not** implemented here yet.
