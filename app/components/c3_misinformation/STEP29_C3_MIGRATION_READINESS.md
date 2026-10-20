# Step 29 — C3 migration readiness

## Scope and database safety

This step prepares the C3 PostgreSQL migration from the repository's current
SQLAlchemy models. No PostgreSQL or Neon database was contacted, provisioned,
modified, upgraded, downgraded, or stamped. The environment has no configured
database URL and no disposable database target.

The branch is `pirakash`; the staging area was empty before and after this
step. Existing unstaged and untracked work was preserved.

## Files created

- `migrations/versions/c3_0001_claim_lifecycle.py` — explicit C3 revision
  creating the claim lifecycle and verification-result tables.
- `STEP29_C3_MIGRATION_READINESS.md` — this readiness report.

No API, repository, worker, ownership, shared-corpus, or authentication code
was changed.

## Revision dependency and graph

The C3 revision declares:

```text
shared_0001 (root, shared branch)
    ↓
c3_0001 (C3 branch head)
```

`c3_0001` uses `down_revision = "shared_0001"` so the shared-corpus baseline
is applied first. It does not alter shared tables and has no foreign key to
shared-corpus data. The migration environment's configured version locations
already include the C3 directory.

The supplied virtualenv does not contain the `alembic` package, so
`alembic heads` could not be executed. The two revision modules were parsed
statically with Python AST checks: revision IDs, down revision, branch labels,
and the complete two-node graph are valid. No Alembic command that executes
DDL (`upgrade`, `downgrade`, or `stamp`) was run.

## Schema coverage

The revision creates schema `c3` and exactly the tables represented by
`app/components/c3_misinformation/models.py`:

### `c3.claims`

- UUID primary key `claim_id`
- input and multilingual fields: `input_type`, `raw_input_text`,
  `cleaned_text`, `detected_language`, `extracted_claim`, `legal_domain`
- lifecycle timestamps and state: `created_at`, `processing_status`,
  `queued_at`, `started_at`, `completed_at`, `failed_at`,
  `failure_category`
- worker fencing: `worker_token`, `lease_expires_at`
- anonymous ownership: `anonymous_session_hash CHAR(64)`
- server defaults for `created_at`, `processing_status='queued'`, and
  `queued_at`
- all model check constraints for input type, nonblank text, lifecycle shape,
  failure categories, lease shape, and lowercase SHA-256 owner hashes
- partial index `claims_anonymous_session_hash_idx` for non-null owner hashes

### `c3.verification_results`

- UUID primary key `result_id`
- unique foreign key `claim_id → c3.claims.claim_id`
- `verdict`, nullable `confidence_score NUMERIC(4,3)`, `explanation`,
  `llm_model_name`, `retrieval_top_k`, JSONB `result_payload`, and
  `created_at`
- model check constraints for verdict, confidence range, top-k range, and
  JSON object payloads

No fields, indexes, relationships, or PostgreSQL types outside the models
were introduced. The migration intentionally does not require pgvector or
modify the shared legal corpus because C3 lifecycle tables do not use vector
columns.

## Upgrade and downgrade safety

Upgrade first rejects pre-existing C3 lifecycle tables rather than adopting or
overwriting unknown objects, then creates only schema `c3` and the two C3
tables. It does not drop or alter shared-corpus tables, extensions, or other
schemas.

Downgrade locks the two C3 tables, refuses to continue if either contains
rows, drops the C3 result table, owner-hash index, and claims table, and leaves
schema `c3` in place. It therefore avoids silent lifecycle/result data loss
and avoids dropping objects that may belong to another revision.

## Offline validation

Passed:

- Python compilation of the new revision
- static revision graph and dependency checks
- static model-to-migration column and constraint coverage checks
- C3 model, repository, and worker lifecycle tests: **33 passed**
- `git diff --check`

Not executed:

- Alembic runtime graph commands, because Alembic is absent from the supplied
  virtualenv
- PostgreSQL upgrade/downgrade execution, because no authorized disposable
  database exists
- live constraint, lock, FK, timestamp, transaction, and ownership tests

Known TestClient/AnyIO hanging tests remain excluded. They are unrelated to
the migration's static validation.

## Risks and remaining requirements

Before applying this revision, the project needs an approved disposable
PostgreSQL database, the Alembic runtime dependency, and an explicit migration
connection. A live validation must confirm schema creation, all constraints,
foreign-key and unique-result behavior, partial-index creation, worker lease
locking (`FOR UPDATE SKIP LOCKED`), stale-worker fencing, rollback behavior,
and safe empty/non-empty downgrade handling.

The existing migration environment is explicitly corpus-oriented and reads
`CORPUS_MIGRATION_URL`; deployment owners must approve whether that connection
and revision chain are also the correct authority for C3. This step does not
change that runtime policy or infer deployment readiness from static files.

## Git state

Branch: `pirakash`
Staging: empty
Database changes executed: none
History/staging commands used: none
