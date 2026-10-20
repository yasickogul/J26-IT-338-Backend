-- C4 migration: create only the dedicated c_argumentation schema.
-- Review and execute through the approved migration process after approval.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS c_argumentation;

CREATE TABLE IF NOT EXISTS c_argumentation.court_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_number TEXT NOT NULL,
    source_document_key TEXT NOT NULL,
    date_decided DATE,
    parties TEXT,
    judge TEXT,
    keywords TEXT,
    legislation TEXT,
    case_type TEXT,
    pdf_url TEXT,
    source_csv TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_c4_court_cases_source_document_key UNIQUE (source_document_key)
);

CREATE TABLE IF NOT EXISTS c_argumentation.case_judgments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES c_argumentation.court_cases(id) ON DELETE CASCADE,
    full_text TEXT,
    page_count INTEGER,
    extraction_status TEXT NOT NULL DEFAULT 'pending',
    extraction_method TEXT,
    extracted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS c_argumentation.judgment_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    judgment_id UUID NOT NULL REFERENCES c_argumentation.case_judgments(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    start_page INTEGER,
    end_page INTEGER,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_c4_judgment_chunks_order UNIQUE (judgment_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS ix_c4_court_cases_date_decided ON c_argumentation.court_cases (date_decided);
CREATE INDEX IF NOT EXISTS ix_c4_court_cases_case_type ON c_argumentation.court_cases (case_type);
CREATE INDEX IF NOT EXISTS ix_c4_court_cases_source_csv ON c_argumentation.court_cases (source_csv);
CREATE INDEX IF NOT EXISTS ix_c4_court_cases_pdf_url ON c_argumentation.court_cases (pdf_url) WHERE pdf_url IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_c4_judgment_chunks_judgment_id ON c_argumentation.judgment_chunks (judgment_id);
