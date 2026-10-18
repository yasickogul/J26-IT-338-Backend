"""Async SQLAlchemy engine + session factory for Neon PostgreSQL."""

from collections.abc import AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings


def normalize_async_database_url(url: str) -> str:
    """Make a Neon / Postgres URL usable with SQLAlchemy + asyncpg.

    Neon "Connect" strings often look like:
      postgresql://...@...-pooler.../neondb?sslmode=require&channel_binding=require
    asyncpg needs:
      postgresql+asyncpg://...@.../neondb?ssl=require
    """
    if not url:
        return url

    raw = url.strip()
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://") :]
    if raw.startswith("postgresql://"):
        raw = "postgresql+asyncpg://" + raw[len("postgresql://") :]

    parsed = urlparse(raw)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))

    # asyncpg uses `ssl`, not `sslmode`; it rejects channel_binding.
    if "sslmode" in query:
        query.setdefault("ssl", query.pop("sslmode"))
    query.pop("channel_binding", None)
    if "ssl" not in query:
        query["ssl"] = "require"

    return urlunparse(parsed._replace(query=urlencode(query)))


def _build_engine() -> AsyncEngine | None:
    if not settings.database_url:
        return None
    return create_async_engine(
        normalize_async_database_url(settings.database_url),
        pool_pre_ping=True,  # Neon autosuspends idle computes
        pool_size=5,
        max_overflow=5,
    )


engine = _build_engine()
SessionLocal = async_sessionmaker(engine, expire_on_commit=False) if engine else None


async def get_session() -> AsyncIterator[AsyncSession]:
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL is not set")
    async with SessionLocal() as session:
        yield session


async def check_connection() -> dict:
    """Run a trivial query to verify the DB is reachable."""
    if engine is None:
        return {"ok": False, "error": "DATABASE_URL is not set"}
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1 AS ok, current_database() AS db, version() AS version"))
            row = result.mappings().one()
            return {
                "ok": True,
                "database": row["db"],
                "version": str(row["version"]).split(",")[0],
            }
    except Exception as exc:  # surface a clean error to /health/db
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
