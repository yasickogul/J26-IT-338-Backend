# J26-IT-338 - Backend - kavy - Civil Case Intelligence


## Overview

The backend of the Civil Case Intelligence system is developed using **Python and FastAPI**.

It converts the original `civil-cases-final` Jupyter Notebook pipeline into web-accessible API services while maintaining the notebook's core processing logic.

The backend processes uploaded Sri Lankan civil case documents and generates structured Civil Case Intelligence Reports containing extracted information, legal issue explanations, relevant historical cases, source citations, party positions, and evidence gaps.

## Main Features

- **Document Processing:** Process uploaded PDF and image documents.
- **Text Extraction:** Extract text using PyMuPDF and EasyOCR.
- **Legal Information Extraction:** Identify and organize relevant information into nine categories.
- **Document Chunking:** Split large documents into manageable text sections.
- **AI-Powered Analysis:** Use an LLM to extract structured information and generate legal explanations.
- **Semantic Retrieval:** Retrieve relevant legal information using sentence-transformers and Qdrant.
- **Retrieval-Augmented Generation (RAG):** Generate legal explanations grounded in retrieved sources.
- **Historical Case Retrieval:** Find similar legal cases from the knowledge base.
- **Evidence Gap Identification:** Identify missing or weak supporting evidence.
- **Party Position Analysis:** Organize the positions of parties involved in the case.
- **Report Generation:** Generate a structured report with legal explanations, source references, and a disclaimer.
- **Document Export:** Generate PDF and Markdown reports.
- **Asynchronous Job Processing:** Process lengthy analysis requests while providing progress updates through API endpoints.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming language |
| FastAPI | REST API development |
| Uvicorn | ASGI application server |
| PyMuPDF | PDF text extraction |
| EasyOCR | OCR for scanned documents and images |
| Sentence Transformers | Text embeddings |
| Qdrant | Vector database for legal document retrieval |
| OpenRouter API | LLM access for AI-based analysis |
| ReportLab | PDF report generation |
| Pandas / Datasets | Dataset processing |

## Backend Project Structure

```text
J26-IT-338-Backend/
    BACKEND_ARCHITECTURE.md
    README.md
    app/components/c2_case_analysis/
        dev_app.py
        router.py
        runtime.py
        config.py
        pipeline.py
        extraction.py
        retrieval.py
        ocr.py
        llm.py
        report.py
        pdf_export.py
        ...other analysis modules
        scripts/build_knowledge_base.py
        data/fewshot.json
        docs/ORIGINAL_PROJECT_README.md
        requirements.txt
        .env.example
        .env                 # local, ignored by Git
        .venv/               # local, ignored by Git

Frontend: ../J26-IT-338-Frontend/ (separate repository, branch kavy)
```

## Backend Processing Workflow

1. **Document Upload:** Receive PDF or image documents through the API.
2. **Text Extraction:** Extract digital text using PyMuPDF, with EasyOCR fallback for image-based pages.
3. **Text Chunking:** Split extracted content into smaller processing units.
4. **Information Extraction:** Use the LLM to identify legal information and classify it into nine categories.
5. **Duplicate Removal:** Merge near-duplicate extracted information using a similarity threshold of 0.85.
6. **Knowledge Retrieval:** Search the Qdrant vector database for relevant cases and legal sources.
7. **RAG Explanation Generation:** Generate legal issue explanations supported by retrieved sources.
8. **Evidence and Party Analysis:** Identify party positions and potential evidence gaps.
9. **Report Assembly:** Combine extracted details, historical case references, citations, and legal explanations.
10. **Report Export:** Generate downloadable PDF and Markdown reports.

## How to Run the Backend

**Step 1: Open the backend directory**

```powershell
cd J26-IT-338-Backend
```

**Step 2: Create a Python virtual environment (Windows)**

```powershell
python -m venv .venv
```

**Step 3: Activate the environment**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Step 4: Install dependencies**

```powershell
python -m pip install -r app/components/c2_case_analysis/requirements.txt
```

**Step 5: Configure environment variables (fresh clones only; preserve your existing .env)**

```powershell
Copy-Item app/components/c2_case_analysis/.env.example app/components/c2_case_analysis/.env
```

Edit `.env` and provide the required credentials:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
QDRANT_API_KEY=your_qdrant_api_key
HF_TOKEN=your_huggingface_token
```

**Step 6: Start the FastAPI server**

```powershell
python -m uvicorn app.components.c2_case_analysis.dev_app:app --reload --port 8000
```

**Step 7: Access the API documentation**

http://127.0.0.1:8000/docs

## Backend API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/c2/analyze` | Upload document and initiate background analysis |
| GET | `/api/v1/c2/health` | Check backend initialization and warmup status |
| GET | `/api/v1/c2/jobs/{id}` | Retrieve job status, stage, progress, and warnings |
| GET | `/api/v1/c2/jobs/{id}/extraction` | Retrieve extracted legal categories |
| GET | `/api/v1/c2/jobs/{id}/result` | Retrieve the complete analysis report |
| GET | `/api/v1/c2/jobs/{id}/report.pdf` | Download the PDF report |
| GET | `/api/v1/c2/jobs/{id}/report.md` | Download the Markdown report |
| DELETE | `/api/v1/c2/jobs/{id}` | Stop a running analysis job |

## Knowledge Base Setup

If the required Qdrant collection does not exist, build the knowledge base by running:

```bash
cd J26-IT-338-Backend
python -m app.components.c2_case_analysis.scripts.build_knowledge_base --plain-dir /path/to/law-docs-054 --scanned-dir /path/to/law-docs-scanned
```

The knowledge base contains processed legal documents that support semantic search and RAG-based explanation generation.

## Initial Model Setup

On the first successful startup, the backend downloads the required embedding model, EasyOCR weights, and dataset for few-shot examples.

The few-shot dataset is cached at:

```text
app/components/c2_case_analysis/data/fewshot.json
```

Check the following endpoint:

```text
GET /api/v1/c2/health
```

A `warmup: ready` response indicates that backend initialization has completed.

## Improvements Over the Original Notebook

### 1. Document Chunking Fix

The original notebook could repeatedly generate overlapping chunks at the end of a document, causing excessive LLM calls.

The updated implementation emits the final chunk only once while preserving earlier chunk boundaries.

### 2. Asynchronous Processing

Document analysis runs as a background job, allowing the frontend to retrieve progress updates using a job identifier.

### 3. Improved Error Handling

The backend exposes errors and warnings relating to unreadable documents, invalid LLM-generated JSON, and unsupported upload formats or sizes.

### 4. Environment-Based Configuration

Sensitive API keys and service configuration are stored in environment variables rather than Kaggle Secrets or hard-coded values.

### 5. Source-Cited Explanations

Generated explanations include numbered citations mapped to retrieved legal sources.

## Backend Limitations

- OCR processing currently supports English.
- The Helvetica-based PDF export does not correctly render Sinhala or Tamil characters.
- Jobs are stored in memory for approximately six hours.
- A single Uvicorn worker is recommended for the current in-memory job implementation.
- Multiple-worker deployment would require shared job storage such as Redis or a database.
- The default `LLM_MODEL` is `openai/gpt-6-luna` according to the project documentation; the model identifier must be verified against OpenRouter's available models.
- The `outcome_prediction` flag is calculated during prompt-injection screening but is not used elsewhere in the original notebook pipeline.

## Backend Responsibility

The backend manages all **document processing, AI analysis, information extraction, knowledge retrieval, report assembly, and report generation**.

It exposes these operations to the React frontend through JSON-based REST API endpoints.