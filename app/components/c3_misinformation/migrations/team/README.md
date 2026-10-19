# C3-only migration proposal for the existing team Neon database

Status: **prepared, not applied**. No stamp or online upgrade was executed.
Target authorized for inspection: LexLanka_App / production / neondb, using the
existing pooled endpoint. Project/branch ownership is the team's confirmation;
SQL catalogs cannot independently prove the Neon Console project/branch name.
The URL and credentials are deliberately absent from this document.

## Observed live schema and compatibility

Read-only catalog inspection confirmed pgvector 0.8.6, 50 documents, 629 chunks,
an existing `c3` schema, no C3 lifecycle tables, and no Alembic version
table. The connected role has USAGE and CREATE privileges on `c3`.
Existing C1 tables are `c1.case_intakes` and `c1.extractions`. No writes were made.

| Contract | Existing database | Repository shared_0001 expectation |
| --- | --- | --- |
| Chunk embedding | nullable `vector(384)` | non-null `vector(768)` |
| Document citation identity | no `source_id` column | nonblank unique `source_id` |
| Chunk citation/provenance | missing `chunk_id`, `original_chunk_id` | both required; chunk_id unique |
| Embedding provenance | missing `embedding_text`, `embedding_model` | both required |
| Parent context | missing `parent_context`, `paragraph` | columns present, nullable |
| Chunk timestamp | missing `created_at` | required with default |
| Chunk document link | nullable FK | non-null FK |
| Metadata | nullable on both tables | non-null; object checks |
| Shared constraints | PKs and document FK; document citation unique index | additional nonblank, JSON, position and source-ID constraints |
| Search indexes | HNSW on 384-vector, GIN text index, document index | model and parent lookup indexes plus document/GIN indexes |

Document `created_at` is also nullable; document language defaults to `en` in
the database, whereas the repository baseline has no language default.
The existing corpus is **not** an applied `shared_0001` baseline. Do not stamp
it as such and do not run the root upgrade chain against this database.
Its guard would refuse the existing tables. Changing or replacing the populated
shared corpus is outside this proposal.

The C3 lifecycle tables have no FK to shared tables and can be installed safely
without corpus reconciliation. Real C3 RAG is still blocked: its current SQL
expects the missing columns and LaBSE 768-dimensional vectors. `/legal-sources`
also selects the absent document `source_id` column. Creating C3 tables will
not fix these retrieval/metadata incompatibilities. A future storage smoke test
must inject deterministic tokenization/RAG dependencies, explicitly label fake
assessment data, and have separate authorization to insert production data.

## Proposed migration and revision ownership

[alembic.ini](./alembic.ini) selects only [versions/](./versions), independent
of the root configuration. The new root revision is `c3_team_0001`, with no
shared dependency. Its upgrade and downgrade DDL are frozen copies of the
existing [c3_0001 lifecycle migration](../versions/c3_0001_claim_lifecycle.py).
Offline tests require the lifecycle SQL to remain identical.

Track this independent lineage in **`c3.alembic_version_c3`**. This is one
additional bookkeeping table, alongside the two requested application tables.
It records `c3_team_0001`; it makes no assertion about shared or C1/C2 history.
Alembic supports independent environments with separate version tables:
https://alembic.sqlalchemy.org/en/latest/cookbook.html#run-multiple-alembic-environments-from-one-ini-file

The team should explicitly adopt this ownership rule before execution:

- C3 schema migrations on this database use this configuration and lineage.
- Other components/shared tables remain under their owners' existing process.
- Future C3 revisions extend `c3_team_0001`; do not apply the old C3 baseline
  too, stamp the shared baseline, or combine histories without a reviewed plan.
- Existing database schemas require their own audited baseline decisions;
  object existence alone is not proof that a migration ran.
- Only one migration operator runs at a time. Autogeneration is intentionally
  not configured here. Review every future revision for schema ownership.

The online environment requires a canonical `C3_MIGRATION_URL` with driver
`postgresql+asyncpg`, database `neondb`, and `ssl=verify-full`; it also requires
`C3_MIGRATION_EXPECTED_HOST` to match the separately approved endpoint exactly.
Do not set that expectation from an arbitrary URL without checking the Console.
Neither variable is automatically read from `.env`. The existing app normalizer
can derive the migration URL locally from DATABASE_URL; use SQLAlchemy's URL
parser to set verify-full and preserve encoded credentials. Never print it,
put it in shell history, or commit it. Keep application configuration unchanged.

Online execution uses TLS certificate/hostname verification, NullPool, disabled
driver/prepared-statement caches, a 5-second lock timeout and 30-second statement
timeout. It requires the existing `c3` schema and refuses a global
`public.alembic_version` table pending review. The transaction encompasses
preflight, C3 DDL, and revision tracking; exceptions roll it back.

## Exact proposed changes

[reviewed_upgrade.sql](./reviewed_upgrade.sql) is the complete first-application
DDL and version insertion, rendered locally with Alembic Operations in offline
mode. No Alembic upgrade command was invoked to prepare it. It contains:

1. `CREATE TABLE c3.alembic_version_c3` with its primary key.
2. A guard rejecting either pre-existing C3 lifecycle relation.
3. `CREATE SCHEMA IF NOT EXISTS c3` (already exists; no shared schema action).
4. `CREATE TABLE c3.claims`: 17 columns, primary key, seven checks covering input,
   status, failures, timestamps, worker lease and owner-hash format.
5. The partial `claims_anonymous_session_hash_idx` index.
6. `CREATE TABLE c3.verification_results`: nine columns, primary key, C3-only FK,
   unique claim ID and four checks for verdict/confidence/top-k/JSON payload.
7. Insert `c3_team_0001` into the C3 version table and commit.

Primary/unique constraints also create PostgreSQL backing indexes. There is no
shared/C1/C2 DDL, extension operation, corpus write, DROP, TRUNCATE or ALTER in
the upgrade. Existing corpus counts are observations, not deployment guarantees.
The environment's read-only catalog preflight and transaction-local timeout
settings precede the DDL shown in the SQL file. Do not apply the SQL file directly;
use the reviewed Alembic environment so its target/preflight checks are retained.

## Commands and approval checkpoint

From the repository root, these inspect files only:

```bash
./.venv/bin/python -m alembic -c app/components/c3_misinformation/migrations/team/alembic.ini heads
./.venv/bin/python -m alembic -c app/components/c3_misinformation/migrations/team/alembic.ini history
PYTHONPATH=. ./.venv/bin/python -m unittest app.components.c3_misinformation.tests.test_team_migration -v
```

**Only after explicit approval**, securely populate the process variables above,
recheck the endpoint and absence of target tables, and run the exact revision:

```bash
./.venv/bin/python -m alembic -c app/components/c3_misinformation/migrations/team/alembic.ini upgrade c3_team_0001
```

Do not run the root configuration, `stamp`, `create_all`, or a downgrade during
installation. If objects already exist or tracking differs, stop and inspect;
do not try to force adoption. Do not log raw connection exceptions containing
credentials or connection strings.

After approval and successful application, run these in a **read-only transaction**:

```sql
BEGIN READ ONLY;
SELECT current_database(), current_setting('transaction_read_only');
SELECT version_num FROM c3.alembic_version_c3;
SELECT schemaname, tablename FROM pg_tables
 WHERE schemaname IN ('c1', 'c2', 'shared', 'c3')
 ORDER BY schemaname, tablename;
SELECT (SELECT count(*) FROM c3.claims) AS claims,
       (SELECT count(*) FROM c3.verification_results) AS results,
       (SELECT count(*) FROM shared.legal_documents) AS documents,
       (SELECT count(*) FROM shared.legal_chunks) AS chunks;
SELECT table_name, column_name, data_type, is_nullable
 FROM information_schema.columns WHERE table_schema = 'c3'
 ORDER BY table_name, ordinal_position;
SELECT c.relname, con.conname, pg_get_constraintdef(con.oid)
 FROM pg_constraint con JOIN pg_class c ON c.oid = con.conrelid
 JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'c3';
SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'c3';
ROLLBACK;
```

Expected: C3 revision `c3_team_0001`, two empty lifecycle tables and the tracking
table; existing shared definitions unchanged. Investigate count differences
against concurrent team activity rather than assuming they prove migration writes.

## Rollback

A failed first application rolls back transactional DDL and revision insertion.
After successful installation, prefer fixing forward. No rollback is authorized.
Before any separately approved downgrade, stop C3 workers/writers, verify a
recoverable backup, inspect FK dependencies, and confirm both C3 tables are empty.
The copied downgrade takes exclusive locks and refuses nonempty tables; it has
no CASCADE and leaves `c3` and its bookkeeping table in place (with no revision
row after downgrade to base). It drops only the two lifecycle tables and their
indexes. It must never be used to discard real claims. Shared tables are untouched.

## Validation

Offline tests cover independent revision discovery, exact lifecycle DDL parity,
schema-local SQL scope, the reviewed SQL artifact, fail-closed URL/preflight
validation, and downgrade refusal intent. These are not live migration execution
tests. The full C3 suite is separately bounded because this agent environment
has previously differed from the team's normal WSL test runtime.
