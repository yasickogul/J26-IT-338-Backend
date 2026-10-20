# Independent C3 legal corpus

## Ownership and safety boundary

C3 owns `c3.legal_sources` and `c3.legal_chunks`. Claim processing continues to
use `c3.claims` and `c3.verification_results`. C3 does not read or write the
team's `shared.legal_documents` or `shared.legal_chunks`, and it has no automatic
fallback to those tables.

The corpus is copied once into this repository and then imported separately
after review. Runtime retrieval uses PostgreSQL rows; the local legal documents
are not required by the worker after a verified import. They remain valuable as
the immutable source material for provenance, revalidation, and disaster
recovery.

## Local copied-artifact layout

The repository ignores `data/`, including every corpus payload. Create and use:

```text
data/c3_misinformation/
  legal_documents/    original reviewed PDF, DOCX, or TXT files
  cleaned_documents/  existing reviewed cleaned documents
  chunks/             existing original corpus/chunk JSON
  embeddings/         prepared-input JSON and LaBSE vector NPZ
  manifests/          existing generation/validation manifest JSON
```

Copy real files into these directories manually. Do not use symlinks, absolute
paths to an old project, or automatic synchronization. The importer never
deletes, moves, cleans, chunks, embeds, or overwrites these files.

## Files required for validation

The current importer deliberately supports the already-reviewed four-artifact
format. Supply all of these explicitly from inside `data/c3_misinformation/`:

1. Original corpus JSON containing the prepared legal chunks and metadata.
2. Prepared embedding-input JSON.
3. Embedding manifest JSON containing the input/corpus hashes, model name,
   dimension, and record count.
4. Vector NPZ containing `embeddings` and ordered `chunk_ids` arrays.
5. Every original source named by chunk `metadata.source_file`, copied beneath
   `legal_documents/` using the same relative name.

The vector artifact must identify `sentence-transformers/LaBSE`, contain finite
unit-normalized `float32` vectors with dimension 768, and have IDs in exactly the
same order as the prepared inputs. Existing chunk boundaries and legal text are
preserved. Unknown formats are rejected rather than transformed or guessed.

Cleaned documents are retained locally but are not consumed by this importer;
the reviewed chunk and embedding artifacts are authoritative for the one-time
database load.

## Offline dry-run validation

The module CLI only validates and cannot connect to PostgreSQL:

```bash
python -B -m app.components.c3_misinformation.corpus_import \
  --manifest manifests/embedding_manifest.json \
  --input embeddings/prepared_inputs.json \
  --corpus chunks/legal_chunks.json \
  --vector embeddings/labse_vectors.npz
```

Names above are examples; retain the copied files' reviewed names. Relative and
absolute arguments are accepted only when they resolve inside the configured C3
data root. Missing files, symlinks, path traversal, hash mismatches, duplicate
IDs, metadata conflicts, malformed citations, non-LaBSE model identity,
incorrect dimensions, non-finite/non-normalized vectors, and ID-order mismatch
all fail validation.

## C3 PostgreSQL schema

`c3.legal_sources` stores stable source IDs, titles, legal domain, document type,
language, citation, optional source URL, a repository-relative source-file
reference, SHA-256 checksum, source metadata, and timestamps.

`c3.legal_chunks` stores stable and original chunk IDs, source relationship,
source-relative index and section, exact legal and embedding text, JSON metadata,
parent context, paragraph, embedding-model provenance, and a non-null
`vector(768)` embedding. It uses exact cosine retrieval; no approximate vector
index is created before recall has been evaluated.

The canonical revision is `c3_0002`, following `c3_0001`. The independently
tracked team-database revision is `c3_team_0002`, following `c3_team_0001`.
These are alternate migration lineages for different deployment histories; do
not apply both. Both revisions refuse existing corpus objects, require a
pre-provisioned pgvector extension, and refuse downgrade when either C3 corpus
table contains data. Neither revision touches `public` or `shared` tables.

Migration application and database import require separate authorization. The
import function additionally requires the exact `IMPORT_C3_CORPUS` authorization
value, revalidates all local artifacts before database access, locks only the two
C3 corpus tables, and refuses any nonempty target. There is no merge, upsert,
truncate, replacement, or duplicate overwrite behavior.

## Retrieval and persistence flow

```text
c3.claims queued text
  -> C3 worker and optional claim extraction
  -> local cached LaBSE query embedding
  -> exact cosine search in c3.legal_chunks
  -> same-source parent expansion and original-ID lookup
  -> evidence packet and citation-constrained RAG prompt
  -> Gemini verification or empty-evidence abstention
  -> immutable result snapshot in c3.verification_results
```

The `/api/v1/c3/legal-sources` endpoint reads `c3.legal_sources`. Result payloads
retain evidence passages and citation IDs. Image OCR remains an extraction-only
workflow and does not create a persisted image claim.

## Current blockers

- The reviewed source documents and four corpus artifacts have not been copied.
- No migration has been applied.
- No database import has been authorized or executed.
- The worker still requires locally provisioned LaBSE tokenizer/model assets and
  runtime packages; no model is downloaded automatically by the configured path.
- Real PostgreSQL/pgvector behavior and retrieval quality require separately
  authorized validation after schema and data installation.
