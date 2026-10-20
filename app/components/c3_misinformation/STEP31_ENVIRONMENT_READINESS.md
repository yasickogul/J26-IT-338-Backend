# Step 31 — Python environment and Alembic dependency readiness

## Current environment

- Repository: `/home/sano_01/J26-IT-338-Backend`
- Branch: `pirakash`
- Project interpreter: `/home/sano_01/fyp/backend/.venv/bin/python`
- Python: `3.12.3`
- SQLAlchemy: `2.0.52`
- FastAPI: `0.141.1`
- Project virtualenv Alembic: **missing**
- Project virtualenv asyncpg: **missing**

The project virtualenv is the correct interpreter for C3 tests and remains
unchanged. No package installation was attempted.

## Dependency declarations and required installation

[requirements.txt](../../../requirements.txt) already declares:

```text
sqlalchemy[asyncio]>=2.0
asyncpg>=0.29
alembic>=1.13
```

[requirements-dev.txt](../../../requirements-dev.txt) includes the base
requirements and therefore needs no edit.

The minimal installation command for the existing virtualenv is:

```text
/home/sano_01/fyp/backend/.venv/bin/python -m pip install "alembic>=1.13" "asyncpg>=0.29"
```

This command was intentionally **not executed**, per the Step 31 restriction.
Reinstalling the complete requirements file is unnecessary for the two
missing packages.

## Compatibility findings

Python 3.12.3 satisfies the supported runtime range of the installed FastAPI,
Pydantic, Starlette, SQLAlchemy, and greenlet packages. SQLAlchemy 2.0.52 is
the correct major version for the migration environment's
`DeclarativeBase`/async engine code. Alembic 1.13+ and asyncpg 0.29+ are the
declared compatible lower bounds; no demonstrated version conflict requires
changing those declarations.

The host has `/usr/bin/alembic` 1.13.1, but the project virtualenv does not.
Using host/site-package mixing is not a final solution and was not added to
the project configuration.

## Offline validation

Passed:

- `pip check` in the project virtualenv: no broken requirements
- migration module import using an explicitly supplied external Alembic
  runtime: `shared_0001` and `c3_0001` loaded with the expected dependencies
- migration syntax compilation
- C3 model/repository/worker lifecycle tests: **33 passed**
- `git diff --check`

The previously validated Alembic heads/history and offline SQL generation
remain valid, but those commands require Alembic outside the project
virtualenv. No database connection or online migration command was used.

## Remaining blockers

1. Install the two already-declared packages into the project virtualenv
   using the command above.
2. Re-run Alembic heads/history and offline SQL generation from that same
   virtualenv.
3. Obtain an explicitly authorized disposable PostgreSQL database before any
   online migration or constraint/locking verification.
4. Install/provision PostgreSQL-side requirements such as asyncpg and the
   shared-corpus pgvector extension only through the approved environment
   process.

No application logic, API contract, migration, model, repository, worker, or
ownership code was changed in Step 31.

## Git status

Branch: `pirakash`  
Staged files: `0`  
Existing staged/unstaged/untracked work: preserved  
Package installation: not performed  
Database access: not attempted
