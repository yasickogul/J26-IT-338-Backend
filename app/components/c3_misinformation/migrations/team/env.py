"""Independent C3 history for an existing, separately managed team database.

No .env loading, shared baseline adoption, autogeneration, or schema bootstrap.
Online execution requires an explicitly supplied URL and approved hostname.
"""
import asyncio
import os
import ssl

from alembic import context
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool


def approved_url():
    raw = os.environ.get("C3_MIGRATION_URL", "")
    expected_host = os.environ.get("C3_MIGRATION_EXPECTED_HOST", "")
    try:
        url = make_url(raw)
        valid = (
            url.drivername == "postgresql+asyncpg"
            and bool(expected_host)
            and url.host == expected_host
            and url.host.endswith(".neon.tech")
            and url.database == "neondb"
            and bool(url.username) and bool(url.password)
            and set(url.query).issubset({"ssl"})
            and url.query.get("ssl", "verify-full") == "verify-full"
        )
    except Exception:
        valid = False
    if not valid:
        raise RuntimeError(
            "Explicit C3 migration URL, approved Neon hostname, neondb, "
            "and verify-full TLS are required. Connection details suppressed."
        ) from None
    return url


def configure(connection=None):
    context.configure(
        connection=connection,
        dialect_name="postgresql" if connection is None else None,
        target_metadata=None,
        version_table="alembic_version_c3",
        version_table_schema="c3",
        literal_binds=connection is None,
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def migrate(connection):
    # These checks run before Alembic can create its version table. Require the
    # pre-existing schema; do not bootstrap or adopt any shared/team objects.
    if connection.scalar(text("SELECT current_database()")) != "neondb":
        raise RuntimeError("Unexpected database; migration refused.")
    if not connection.scalar(text(
        "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'c3')"
    )):
        raise RuntimeError("Existing c3 schema is required; migration refused.")
    # Fail rather than competing with the original global migration lineage.
    if connection.scalar(text("SELECT to_regclass('public.alembic_version')")):
        raise RuntimeError("Global Alembic tracking exists; reconcile ownership first.")
    configure(connection)


async def online():
    engine = create_async_engine(
        approved_url(), poolclass=NullPool,
        connect_args={
            "ssl": ssl.create_default_context(),
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            "timeout": 15,
            "command_timeout": 30,
        },
        hide_parameters=True,
    )
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SET LOCAL lock_timeout = '5s'"))
            await connection.execute(text("SET LOCAL statement_timeout = '30s'"))
            await connection.run_sync(migrate)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    configure()
else:
    asyncio.run(online())
