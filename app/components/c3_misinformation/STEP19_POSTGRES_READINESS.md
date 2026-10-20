# Step 19 PostgreSQL readiness

Update: shared models, migration baseline, validated importer and C3 shared
mapping are now implemented. See [current shared corpus readiness](../../shared/corpus/README.md)
via the repository shared corpus documentation. The sections below describe
the original public-contract baseline; its historical shared-schema blockers
are superseded by the new implementation. Deployment, dependency availability
and actual PostgreSQL verification remain pending.

The three new postgres_* modules are **unconnected** implementations of the
original CompNet migration 001 contract, not the proposed group shared schema.
No table has been deployed or queried as part of this work.

## Supported storage contract

Only explicit selection of `compnet-migration-001-public-legal-chunks` is
accepted. This selector is not proof of approval or deployment verification.
There is no dynamic table substitution or fallback to the shared UUID schema.

The supported relation is `public.legal_chunks`, with:

| Column | Required original contract |
| --- | --- |
| chunk_id | text primary key, nonblank |
| original_chunk_id | text not null, nonblank |
| text | text not null, nonblank; complete legal evidence |
| embedding_text | text not null, nonblank; embedding input provenance |
| metadata | jsonb not null; object for usable evidence |
| parent_context | nullable jsonb object |
| paragraph | nullable text |
| embedding_model | text not null, nonblank |
| embedding | vector(768) not null |
| created_at | timestamptz not null, CURRENT_TIMESTAMP default |

## Mandatory checks before any production wiring

The lead/shared corpus owner must approve the storage location and migration
contract. Verify deployed columns, constraints, pgvector availability, corpus
access permissions and read-only transaction handling. Do not infer deployment
from architecture documentation or the contract selector.

Verify stored embedding_model and query encoder provenance independently:
sentence-transformers/LaBSE, 768 dimensions, normalized embeddings and compatible
tokenization/input preparation. Model-name declaration alone cannot authenticate
an injected encoder. Check the reviewed 348-vector artifact IDs, ordering,
normalization, finite values and input/corpus hashes against the manifest and
actual stored rows. The local manifest is not evidence of a group database import.

Audit exact source_id, legal_domain and language values (English is not en);
source_id and section_label support exact parent lookup with the " > " hierarchy.
Citation metadata includes title, act_number, source_file, language, legal_domain,
document_type and section_label. Retain original chunk IDs and full passage text.
Missing optional citation keys remain missing/null as in source assembly; never
fabricate them. Establish stable globally unique textual IDs and a consistent
corpus view for all three reads.

The group currently documents shared UUID chunks, without explicit original ID,
embedding text, embedding model, parent context or paragraph columns. Shared
models are TODO, ingestion is empty, migrations contain only a placeholder, and
the Alembic environment is missing. Those team-owned decisions remain blockers
to live integration. These modules do not authorize duplicate corpus storage.

## Injection and transaction ownership

After approval/verification, callers may bind session and corpus_contract with
functools.partial. Vector search additionally requires query_model to be bound
from trusted encoder configuration. The resulting callables fit AsyncVectorSearch,
AsyncParentLookup and AsyncOriginalChunkLookup respectively.

The caller owns session creation, a read-only transaction and cleanup; dependencies
only await SELECT statements, without begin/commit/rollback/close. Use the same
consistent corpus view for retrieval, parent expansion and provenance. No runtime
wiring was added. Async embedding must be supplied separately; synchronous model
inference must not run directly on the event loop.

## Semantics and explicit differences

Vector SQL matches source: LaBSE model predicate, optional exact metadata filters,
cosine distance, similarity = 1 - distance, ascending distance then textual
chunk_id, top_k 1..50 (default 5). No approximate index or hybrid scores introduced.
An eventual approximate search plan must be checked against exact source results.

Parent SQL returns all exact source/section matches ordered by chunk_id without a
limit. No extra embedding-model filter is added: source parent SQL has none.
Provenance uses scalar_one: missing and duplicate rows raise, with no fallback.
It accepts only chunk_id because that is the existing evidence-builder interface;
same-source enforcement belongs to parent selection and evidence assembly.

New defensive checks reject malformed returned passage objects and blank original
IDs earlier than source assembly. Missing required columns raise; storage errors
propagate. Invalid non-real vectors receive the source shape/finite error instead
of relying on incidental NumPy errors. Valid source outputs are unchanged.

Existing LaBSE module audit (not changed):

- Missing tokenizer now raises RuntimeError instead of source AttributeError:
  compatibility deviation requiring an explicit follow-up decision/correction.
- Per-call dimension and tokenizer-limit validation are added safeguards, not
  exact source error behavior; they do not prove model identity.
- Injected encoder identity is trusted: a production wiring requirement, not
  independent provenance verification.
- Loader is lazy but not explicitly cache-only, as in source embeddings.py.
  The tokenizer-only admission module is cache-only. Do not claim embedding
  loading is offline-only; a cache-only policy needs a separately approved change.

## Live checks still required

Offline SQL compilation/mocks do not establish asyncpg vector casting, PostgreSQL
operator behavior, real filtering/collation/tie ordering, read-only permissions,
transaction isolation, row visibility, index recall or data provenance.
Run approved disposable PostgreSQL tests, compare representative source queries,
and verify missing/ambiguous parents and original-ID lookup failures before wiring
any worker/RAG path. No live database was used and Step 20 was not started.
