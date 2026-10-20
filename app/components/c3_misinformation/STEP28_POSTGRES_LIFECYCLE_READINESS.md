# Step 28 — PostgreSQL lifecycle readiness

## Scope and safety

This step audited the C3 PostgreSQL lifecycle without contacting a production
database. No database URL, PostgreSQL client, local PostgreSQL listener, or
disposable test instance was available in the execution environment. No
service was started, no migration was applied, and no database data was read
or written.

The working tree was preserved as found. The branch is `pirakash` and the
staging area was empty at inspection time.

## Architecture reviewed

- `app/core/database.py` exposes the existing async SQLAlchemy engine/session
  only when an explicit `DATABASE_URL` is configured. It normalizes async
  PostgreSQL URLs and provides connection diagnostics, but does not create a
  database in tests.
- `app/components/c3_misinformation/models.py` defines the C3 `claims` and
  `verification_results` tables in schema `c3`, including lifecycle, lease,
  ownership, result-shape, and failure-category constraints.
- `repository.py` inserts queued claims, reads owner-scoped state/results,
  acquires one queued text claim with `FOR UPDATE SKIP LOCKED`, renews a live
  lease, and fences extracted/completed/failed writes by worker token and
  expiry. Completion inserts the result snapshot and clears the lease in one
  transaction; duplicate/stale writes therefore cannot overwrite a completed
  claim.
- `claim_worker.py` recovers expired leases, acquires one claim, runs injected
  analysis, and records a completed or categorized failed outcome. Ownership
  loss is handled as a non-owner outcome and does not write a result.
- `api.py` preserves the existing 202 submission contract and owner-scoped
  result endpoint. Cookies are issued only after the submission transaction
  succeeds; foreign or missing owners receive the existing non-disclosing
  result behavior.

## Migration status

The repository contains `alembic.ini` and a migration environment, but the
configured version locations contain only the shared-corpus revision
`shared_0001_corpus.py`; the C3 migration directory contains only `.gitkeep`.
The migration environment targets the shared corpus metadata and requires an
explicit `CORPUS_MIGRATION_URL`. Consequently, there is no reviewed C3
Alembic revision that can create `c3.claims` and
`c3.verification_results` in a disposable database during this step.

This is a schema/deployment blocker, not permission to invent a migration or
apply DDL to an unknown database. A future live run needs an approved,
isolated PostgreSQL database, the C3 migration ownership decision, pgvector
and shared-corpus migrations as applicable, and an explicit migration URL.

## Lifecycle verification status

The requested real-database sequence (submit → queued → leased/processing →
mocked deterministic analysis → completed result → owner retrieval and
foreign-owner denial → safe failure → stale-writer fencing) was **not run**:
there was no isolated disposable PostgreSQL target and no C3 migration to
apply. No claim or result rows were created.

The equivalent offline contracts do pass with mocked sessions and providers:

- queued insertion and owner-scoped reads
- lease acquisition, renewal, expiry recovery, and worker-token fencing
- atomic completion/result snapshot intent and duplicate-write rejection path
- categorized failure and ownership-loss handling
- async HTTP submission/result behavior through `httpx.ASGITransport`
- original claim text, lifecycle states, evidence metadata, and citation IDs
  remain part of the existing contracts

These tests validate SQL construction and transaction intent; they do not
prove PostgreSQL execution, row locks, timestamp behavior, constraints,
permissions, or serialization against a live schema.

## Tests executed

Successful bounded offline runs (Python 3.12.3 from the project virtualenv):

- async HTTP integration: **7 tests passed**
- repository worker lifecycle: **14 tests passed**
- claim worker and runner: **23 tests passed**
- lifecycle/worker contracts: **18 tests passed**
- Step 25 evaluation: **15 tests passed**
- shared corpus: **10 tests passed**
- C3 syntax compilation, imports, and OpenAPI route registration: passed
- `git diff --check`: passed

The focused `test_rag_service_async` plus `test_worker_rag_adapter` command
was bounded at 90 seconds and exited **124** during interpreter shutdown
(after a RAG test body had completed). The focused
`test_worker_dependencies` command was bounded at 60 seconds and also exited
**124** without a completed report. These are recorded as blocked, not passed,
and match the previously documented asyncio/executor shutdown behavior. The
original TestClient/AnyIO-based modules remain intentionally unrerun;
their independently reproduced environment hang is documented in the Step 26
and Step 27 readiness reports. They are not counted as passed here. The full
C3 discovery suite is therefore not claimed as complete.

## Security, concurrency, and production limitations

The repository's worker lease fencing and owner-scoped query predicates are
covered offline, but live PostgreSQL lock and isolation behavior remains
unverified. Before production use, verify schema deployment, constraints,
indexes, asyncpg behavior, transaction isolation, lease expiry under clock
skew, and result serialization with approved test data. Run a deterministic
mock-provider vertical slice only against a disposable database, then test
worker cancellation and stale-owner races. Do not use production data for
these checks.

The existing C3 worker also still requires the configured async session
factory, approved shared legal corpus, cached LaBSE weights, and a configured
provider at runtime. Translation and legal accuracy remain unverified by the
offline mocks.

## Step 27 handoff

Step 27's async HTTP approach is safe to continue using. Step 29 may proceed
with offline/evaluation work, but a live PostgreSQL lifecycle claim must remain
explicitly blocked until an isolated disposable database and an approved C3
migration path are supplied. This limitation must not be hidden by replacing
the original TestClient suite or by reporting SQL-mock tests as end-to-end
database verification.
