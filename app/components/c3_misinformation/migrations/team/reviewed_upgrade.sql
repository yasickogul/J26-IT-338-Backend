BEGIN;

CREATE TABLE c3.alembic_version_c3 (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_c3_pkc PRIMARY KEY (version_num)
);

DO $$ BEGIN
        IF to_regclass('c3.claims') IS NOT NULL
           OR to_regclass('c3.verification_results') IS NOT NULL THEN
            RAISE EXCEPTION 'Existing C3 objects require explicit reconciliation';
        END IF;
    END $$;

CREATE SCHEMA IF NOT EXISTS c3;

CREATE TABLE c3.claims (
    claim_id UUID NOT NULL, 
    input_type VARCHAR(10) NOT NULL, 
    raw_input_text TEXT, 
    cleaned_text TEXT, 
    detected_language VARCHAR(10), 
    extracted_claim TEXT, 
    legal_domain VARCHAR(30), 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, 
    processing_status VARCHAR(16) DEFAULT 'queued' NOT NULL, 
    queued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, 
    started_at TIMESTAMP WITH TIME ZONE, 
    completed_at TIMESTAMP WITH TIME ZONE, 
    failed_at TIMESTAMP WITH TIME ZONE, 
    failure_category VARCHAR(32), 
    worker_token UUID, 
    lease_expires_at TIMESTAMP WITH TIME ZONE, 
    anonymous_session_hash CHAR(64), 
    PRIMARY KEY (claim_id), 
    CONSTRAINT claims_input_type CHECK (input_type IN ('text', 'image')), 
    CONSTRAINT claims_text_raw_input_nonblank CHECK (input_type <> 'text' OR (raw_input_text IS NOT NULL AND raw_input_text !~ '^[[:space:]]*$')), 
    CONSTRAINT claims_processing_status_check CHECK (processing_status IN ('queued', 'processing', 'completed', 'failed')), 
    CONSTRAINT claims_failure_category_check CHECK (failure_category IS NULL OR failure_category IN ('input', 'input_too_long', 'request_too_large', 'retrieval', 'preparation', 'provider', 'response_validation', 'service', 'result_storage', 'interrupted')), 
    CONSTRAINT claims_lifecycle_shape_check CHECK ((processing_status = 'queued' AND started_at IS NULL AND completed_at IS NULL AND failed_at IS NULL AND failure_category IS NULL) OR (processing_status = 'processing' AND started_at IS NOT NULL AND completed_at IS NULL AND failed_at IS NULL AND failure_category IS NULL) OR (processing_status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL AND failed_at IS NULL AND failure_category IS NULL) OR (processing_status = 'failed' AND completed_at IS NULL AND failed_at IS NOT NULL AND failure_category IS NOT NULL)), 
    CONSTRAINT claims_worker_lease_shape_check CHECK ((processing_status = 'processing' AND ((worker_token IS NULL AND lease_expires_at IS NULL) OR (worker_token IS NOT NULL AND lease_expires_at IS NOT NULL))) OR (processing_status <> 'processing' AND worker_token IS NULL AND lease_expires_at IS NULL)), 
    CONSTRAINT claims_anonymous_session_hash_format CHECK (anonymous_session_hash IS NULL OR anonymous_session_hash ~ '^[0-9a-f]{64}$')
);

CREATE INDEX claims_anonymous_session_hash_idx ON c3.claims (anonymous_session_hash) WHERE anonymous_session_hash IS NOT NULL;

CREATE TABLE c3.verification_results (
    result_id UUID NOT NULL, 
    claim_id UUID NOT NULL, 
    verdict VARCHAR(32) NOT NULL, 
    confidence_score NUMERIC(4, 3), 
    explanation TEXT, 
    llm_model_name VARCHAR(100), 
    retrieval_top_k SMALLINT NOT NULL, 
    result_payload JSONB NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL, 
    PRIMARY KEY (result_id), 
    CONSTRAINT verification_results_claim_id_fkey FOREIGN KEY(claim_id) REFERENCES c3.claims (claim_id), 
    CONSTRAINT verification_results_claim_id_key UNIQUE (claim_id), 
    CONSTRAINT verification_results_verdict_check CHECK (verdict IN ('True', 'False', 'Misleading', 'insufficient_evidence')), 
    CONSTRAINT verification_results_confidence_check CHECK (confidence_score BETWEEN 0 AND 1), 
    CONSTRAINT verification_results_top_k_check CHECK (retrieval_top_k BETWEEN 1 AND 50), 
    CONSTRAINT verification_results_payload_object_check CHECK (jsonb_typeof(result_payload) = 'object')
);

INSERT INTO c3.alembic_version_c3 (version_num) VALUES ('c3_team_0001') RETURNING c3.alembic_version_c3.version_num;

COMMIT;

