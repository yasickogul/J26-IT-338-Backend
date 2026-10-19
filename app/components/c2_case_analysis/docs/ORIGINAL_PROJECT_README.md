This is the original import documentation. Its old folder paths and API URLs are historical; use the repository and C2 component READMEs for current commands.

# Civil Case Intelligence (web app)

Web version of the `civil-cases-final` notebook: upload a Sri Lankan civil case document (PDF or image) and get
a **Civil Case Intelligence Report** - parties, key facts, party positions, evidence gaps, and for every legal issue
a source-cited explanation with similar historical cases. Exports to PDF / Markdown / JSON.

```
frontend/   React (Vite)            -> upload, live progress, report, extracted details, downloads
backend/    FastAPI (Python)        -> the notebook's pipeline, unchanged in logic
```

The backend is Python because the notebook's logic depends on PyMuPDF, EasyOCR, sentence-transformers, Qdrant and
ReportLab. The React app talks to it over a small JSON API.

## Run it

### 1. Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then fill in the three secrets
uvicorn app.main:app --reload --port 8000
```
`.env` needs `OPENROUTER_API_KEY`, `QDRANT_API_KEY` and `HF_TOKEN` (the notebook read these from Kaggle Secrets).
On first start the backend downloads the embedding model, EasyOCR weights and the dataset used for few-shot
examples (cached in `backend/data/fewshot.json`). `GET /api/health` shows `warmup: ready` when it is done.

### 2. Frontend
```bash
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api to http://localhost:8000)
```
For production set `VITE_API_BASE_URL` to the backend's public URL and add the frontend origin to `CORS_ORIGINS`.

### 3. Knowledge base (only if the Qdrant collection does not exist yet)
```bash
cd backend
python -m scripts.build_knowledge_base --plain-dir /path/to/law-docs-054 --scanned-dir /path/to/law-docs-scanned
```

## Notebook -> web app map

| Notebook cell | Where it lives now |
|---|---|
| 2-4 dataset, `is_civil`, few-shot sampling (seed 42) | `app/utils.py`, `app/fewshot.py` |
| 5, 13 PDF/image text + EasyOCR fallback (< 20 chars/page, 200 dpi) | `app/ocr.py` |
| 6, 14 chunking | `app/chunking.py` |
| 7 category definitions, extraction prompt, JSON parsing | `app/extraction.py`, `app/llm.py` |
| 8 near-duplicate merge (0.85) | `app/extraction.py` |
| 9 unified output, `run_pipeline` | `app/extraction.py`, `app/pipeline.py` |
| 11, 12 Qdrant connection, embedding model by dimension, BGE prefix | `app/retrieval.py` |
| 15-17 knowledge-base build | `scripts/build_knowledge_base.py` |
| 18 prompt-injection screening | `app/safety.py` |
| 19 retrieval (top 3 cases, top 5 law, min score 0.35) | `app/retrieval.py` |
| 20 grounded RAG explanation + groundedness check | `app/rag.py` |
| 21 party positions + evidence gaps | `app/insights.py` |
| 22 report assembly + disclaimer | `app/report.py` |
| 23 Markdown report | `app/report.py`, "Markdown" download |
| 26 branded PDF | `app/pdf_export.py`, "Download PDF" |

## API

| Method | Path | |
|---|---|---|
| POST | `/api/analyze` | multipart `file` -> `{job_id}` (analysis runs in the background) |
| GET | `/api/jobs/{id}` | status, stage, progress, warnings |
| GET | `/api/jobs/{id}/extraction` | the nine extracted categories (available before the report finishes) |
| GET | `/api/jobs/{id}/result` | extraction + report + markdown |
| GET | `/api/jobs/{id}/report.pdf` / `report.md` | downloads |
| DELETE | `/api/jobs/{id}` | stop a running job |

## Differences from the notebook (deliberate)

1. **Chunking bug fixed.** In the notebook, once the window reached the end of the text the loop advanced one
   character at a time and re-emitted near-identical chunks: a 30,000-character document produced ~580 chunks, i.e.
   ~580 LLM calls. The final window is now emitted once. All earlier chunk boundaries and overlaps are identical.
   If your existing Qdrant collection was built with the old `chunk_for_retrieval`, it may hold many near-duplicate
   tail chunks; re-running the build script cleans that up.
2. **Asynchronous jobs.** Analysis takes minutes, so the API returns a job id and the UI polls. Jobs are kept in
   memory (6 h), so run a single uvicorn worker; use Redis/a DB if you need multiple workers.
3. **Failures are surfaced** instead of silently continuing: unreadable text, a chunk whose model output is not valid
   JSON (shown as a warning), and upload type/size limits.
4. **Secrets and URLs** come from `.env`, not Kaggle Secrets or hard-coded values.
5. **Citations are clickable.** `[n]` in an explanation maps to the n-th entry of "Sources consulted" (the notebook
   builds the context in that same order).

## Known limits (inherited from the notebook)

- `screen_text` also computes an `outcome_prediction` flag, but the notebook never uses it afterwards; kept as is.
- OCR is English only, and the PDF export uses Helvetica, so Sinhala/Tamil text will not render in the PDF.
- `LLM_MODEL` defaults to `openai/gpt-6-luna`; the notebook says to confirm the exact slug on openrouter.ai/models.
