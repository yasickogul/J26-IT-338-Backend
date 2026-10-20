"""Explicit migration connection only; never import app settings or .env."""
import asyncio
import os
from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from app.shared.corpus.models import Base


def configure(connection=None):
    context.configure(
        connection=connection, dialect_name="postgresql" if connection is None else None,
        target_metadata=Base.metadata, include_schemas=True,
        literal_binds=connection is None,
        include_name=lambda name, type_, parent: (
            name == "shared" if type_ == "schema" else
            name in {"legal_documents", "legal_chunks"} if type_ == "table" else True
        ),
    )
    with context.begin_transaction():
        context.run_migrations()


async def online():
    url = os.environ.get("CORPUS_MIGRATION_URL")
    if not url:
        raise RuntimeError("Explicit CORPUS_MIGRATION_URL is required")
    if not url.startswith("postgresql+asyncpg://"):
        raise ValueError("CORPUS_MIGRATION_URL must use postgresql+asyncpg")
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(configure)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    configure()
else:
    asyncio.run(online())
