# Shared legal corpus — Step 19

Status: code implemented; migrations have NOT been applied, corpus has NOT been
imported, and PostgreSQL execution is NOT verified.

## Schema and embedding policy

The ORM in [models.py](models.py) and frozen revision
[shared_0001](../migrations/versions/shared_0001_corpus.py) define:

- shared.legal_documents: UUID id; unique textual source_id; source_type, title,
  citation, court, case_type, decision_date, language, source_url, full_text,
  JSONB metadata and timestamp.
- shared.legal_chunks: UUID id/document_id FK, unique nonblank textual chunk_id,
  nonblank original_chunk_id, per-document chunk_index, section, complete text,
  embedding_text, JSONB metadata/parent_context, paragraph, embedding_model,
  VECTOR(768), generated English tsvector and timestamp.
- Unique document positions; document/model B-tree indexes, source/section
  expression index, and tsvector GIN index. No approximate vector index until
  recall against the original exact ranking is verified.

UUIDs are internal entities. C3 evidence IDs remain "chunk:" + textual chunk_id;
UUIDs never substitute for citation IDs or retrieval tie-breaks. Original IDs
are not foreign keys: the original passage split into children need not have
its own stored embedding row.

One model/vector per textual chunk is supported. embedding_model is mandatory;
C3 selects LaBSE exclusively. A separate embeddings table would support several
models for the same chunk but would change the documented shared contract.
That is deferred; duplicate chunks for other models are prohibited by uniqueness.
Other consumers must use the model that produced the corpus they query.
No change was made to HybridRetriever, RetrievedChunk or core embedding settings.

The SQLAlchemy Vector768 type declares native PostgreSQL VECTOR(768) DDL.
It deliberately provides no ORM vector codec: ingestion uses parameterized
CAST(:embedding AS vector), and retrieval selects evidence fields only.
The pgvector server extension is still required. The Python pgvector package
is not required by these explicit SQL paths.

## C3 mapping and wiring

| C3/source field | Shared mapping |
| --- | --- |
| chunk_id | legal_chunks.chunk_id (text), NOT id (UUID) |
| original_chunk_id | legal_chunks.original_chunk_id |
| source_id/domain/language/section_label | exact keys in chunk metadata |
| text/embedding_text | retained separately without rechunking |
| parent_context/paragraph | same nullable values |
| embedding_model/embedding | explicit model name and 768-vector |
| document identity | document_id UUID links to source_id document |

Chunk metadata remains authoritative for C3; document fields are indexing and
shared presentation aids, not replacements. Import maps Principal Enactment and
Amendment Act to source_type=act while preserving original document_type metadata.
Document full_text is not reconstructed from overlapping chunks.

Use C3 build_shared_evidence_packet with an injected async embedder, trusted
query_model, and caller-owned session/READ ONLY transaction. Shared mapping is
selected explicitly; the original public mapping remains compatible for old tests.
Query vectors must be finite, 768-dimensional and unit-normalized. Model-name
declaration does not independently verify encoder identity.
All reads use the same caller session. For consistency under concurrent ingestion
or updates use an appropriate read-only repeatable-read transaction.
No workers/routes were enabled or production connections created.

## Migrations (manual, separate authorization required)

Provision a dedicated PostgreSQL database and pgvector extension first (in the
normal type search path, typically public). PostgreSQL must support gen_random_uuid.
The available development interpreter currently lacks alembic and asyncpg; use an
environment with the repository's already-declared dependencies installed.
Nothing was installed during this task.

Offline review, when Alembic is available:

    python -m alembic upgrade shared@head --sql

For an explicitly authorized isolated database only, set CORPUS_MIGRATION_URL
securely to a postgresql+asyncpg URL and run:

    python -m alembic upgrade shared@head

No .env/application DATABASE_URL is loaded by the migration environment.
The revision refuses existing corpus objects instead of adopting or altering
them. Reconcile pre-existing objects manually in a reviewed follow-up migration.
The environment lists all documented component version directories but only
shared corpus metadata is included in autogeneration; component migrations
and baselines are not created by this task.

Downgrade shared_0001 to base requires both corpus tables to be empty; it locks
and refuses nonempty tables. It drops only these two empty tables, without
CASCADE, leaving schema and extension in place. Never delete live corpus data
just to satisfy this guard.

## Ingestion

Run dry validation without any database configuration:

    python -B -m app.shared.corpus.ingest --manifest PATH_TO_MANIFEST --input PATH_TO_INPUTS --corpus PATH_TO_CHUNKS --vector PATH_TO_NPZ

All four paths are explicit; paths embedded in the manifest are not followed.
Checks include model/dimension, manifest hashes, exact reviewed input derivation
(including CON-001-0004-a/b), text/metadata correspondence, unique identifiers,
vector ID order, float32 shape, finite values, normalization and citation fields.
The manifest lacks a vector-file hash: validation reports a computed SHA-256
for review but cannot prove that vectors were generated from those inputs merely
from dimensions/norms. Preserve and review the approved artifact hash separately.

The reviewed source artifacts validated as 9 documents / 348 chunks:
vector SHA-256 37bf00d5cb185924e3aca13911396066531996200009637b9c6c2cd69e703e4e.
This is NOT a database import receipt.

CLI never writes to a database. After separate authorization, ingestion tooling
can call import_artifacts(session=dedicated_idle_session, **paths). It revalidates,
locks both tables, rejects ANY existing data, and inserts documents/chunks in
one transaction. No upsert, deletion, overwrite or incremental merging occurs.
Session cleanup is the caller's responsibility. Unique constraints protect
against duplicates; failed inserts roll back the transaction.

## Verification still required

Offline tests compare ORM-generated SQL to frozen migration SQL and test actual
revision functions with an inert op object. This is not an Alembic execution test.
The current environment lacks server binaries and Docker access, plus Alembic and
asyncpg. No PostgreSQL integration test ran.

Provide a fresh disposable local PostgreSQL+pgvector database, a test-only role,
and an environment containing Alembic/asyncpg. Apply shared_0001, inspect pg_catalog
against ORM metadata, insert synthetic unit vectors, then run actual shared C3
queries. Verify filters, exact cosine/tie order, parent ambiguity, provenance
errors, read-only enforcement and import rollback/duplicate prevention. Test
guarded downgrade on empty tables and rejection on populated tables. Cleanup
must target only that disposable database.

The earlier C3 embedding audit still applies: changed missing-tokenizer exception,
extra validation and trusted injected model identity need a separate compatibility
decision; the loader is not explicitly cache-only. No existing embedding code
was changed here. Step 20 has not started.
