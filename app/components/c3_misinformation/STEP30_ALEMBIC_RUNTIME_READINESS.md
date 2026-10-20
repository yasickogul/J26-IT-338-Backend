# Step 30 — Alembic runtime readiness

## Environment inspection

- Repository: `/home/sano_01/J26-IT-338-Backend`
- Branch: `pirakash`
- Project interpreter: `/home/sano_01/fyp/backend/.venv/bin/python`
- Project Python: 3.12.3
- Project SQLAlchemy: 2.0.52
- Project virtualenv Alembic: **not installed**
- Project virtualenv asyncpg: **not installed**
- PostgreSQL connection variables: unset

The dependency declaration is already correct: [requirements.txt](../../../requirements.txt)
contains `alembic>=1.13` and `asyncpg>=0.29`. No dependency file was changed
and no package was installed.

The host also provides `/usr/bin/alembic` 1.13.1. Its system SQLAlchemy is
1.4.50, which is incompatible with this repository's SQLAlchemy 2 metadata
(`DeclarativeBase`). A read-only mixed interpreter path using the project
SQLAlchemy 2.0 site-packages and the host Alembic package was therefore used
for runtime discovery and offline SQL generation. This did not contact a
database.

## Migration discovery

The existing [alembic.ini](../../../alembic.ini) points to `migrations/` and
includes the shared and C3 version directories. Host Alembic discovery with
the compatible mixed path reported:

```text
<base> -> shared_0001
shared_0001 -> c3_0001 (head)
```

`c3_0001` is the single discovered head and correctly declares
`down_revision = "shared_0001"`; branch labels are preserved. The migration
environment imports shared metadata but does not need mutable C3 model
metadata for these explicit revision scripts. Both revision files were
loaded successfully by Alembic.

## Offline SQL validation

The safe command equivalent to:

```text
alembic -c alembic.ini upgrade base:c3_0001 --sql
```

was executed with the compatible mixed runtime. It exited `0` and generated
SQL for both revisions, including:

- `shared.legal_documents` and `shared.legal_chunks` baseline DDL
- `c3.claims` and `c3.verification_results` DDL
- C3 lifecycle, lease, ownership, failure, verdict, confidence, top-k, JSONB,
  FK, uniqueness, defaults, and partial-index definitions
- `alembic_version` transitions from `shared_0001` to `c3_0001`

This was SQL generation only. No SQL was sent to PostgreSQL, and no
`upgrade`, `downgrade`, or `stamp` operation was executed online.

The initial attempt with `/usr/bin/alembic` alone failed before migration
execution because its system SQLAlchemy 1.4 lacked `sqlalchemy.orm.DeclarativeBase`.
The project configuration itself was not changed; the compatible runtime path
resolved the tooling mismatch.

## Schema and model validation

[c3_0001_claim_lifecycle.py](./migrations/versions/c3_0001_claim_lifecycle.py)
was checked against [models.py](./models.py):

- all model columns and PostgreSQL types are represented;
- all named lifecycle, lease, ownership-format, failure, result, and JSONB
  constraints are represented;
- the owner-hash partial index and unique result-per-claim relationship are
  represented;
- the revision creates only schema `c3` objects and does not alter shared
  corpus tables.

Python compilation, static revision checks, migration safety checks, and the
focused model/repository/worker suite all passed.

## PostgreSQL-specific items still unverified

Offline SQL cannot prove execution of PostgreSQL regex checks, `jsonb_typeof`,
`NUMERIC(4,3)` behavior, UUID/FK enforcement, partial indexes, `LOCK TABLE`,
`FOR UPDATE SKIP LOCKED`, lease timestamp semantics, transaction rollback, or
existing-object guards. The shared revision also requires pgvector and uses
generated TSVECTOR expressions. These require an approved disposable
PostgreSQL database.

## Checks

Passed:

- Alembic revision discovery, heads, and history with the compatible runtime
- full-chain offline SQL generation (exit 0)
- migration syntax compilation
- static model-to-migration coverage and safety checks
- C3 model/repository/worker tests: **33 passed**
- `git diff --check`

Blocked or unexecuted:

- Project virtualenv Alembic invocation: blocked by missing package
- Project virtualenv asyncpg invocation: blocked by missing package
- Online migration execution and PostgreSQL integration: intentionally
  unexecuted; no authorized database exists
- Known TestClient/AnyIO hanging tests: intentionally excluded

## Git state

Branch: `pirakash`  
Staged files: `0`  
Migration/configuration files changed in this step: documentation only; the
Step 29 migration was preserved unchanged.  
No reset, restore, clean, stage, commit, merge, push, upgrade, downgrade, or
stamp command was used.
