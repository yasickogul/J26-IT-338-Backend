"""Shared fixtures for offline FastAPI tests (HTTPX ASGI + mocked externals)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_check_connection() -> Iterator[AsyncMock]:
    """Prevent main app lifespan from opening a real database connection."""
    mock = AsyncMock(return_value={"ok": False, "error": "database disabled in automated tests"})
    with patch("app.main.check_connection", mock):
        yield mock


@pytest_asyncio.fixture
async def app(mock_check_connection: AsyncMock):
    """Import the FastAPI app only after Neon URLs have been cleared by root conftest."""
    from app.main import app as fastapi_app

    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    """HTTPX ASGI client with FastAPI lifespan startup/shutdown.

    httpx 0.28's ASGITransport has no `lifespan=` kwarg; use Starlette's
    lifespan_context so startup hooks (e.g. DB check) still run under mocks.
    """
    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test") as ac,
    ):
        yield ac


@pytest_asyncio.fixture
async def c1_app():
    from app.components.c1_CaseAnalysis.dev_app import app as c1_fastapi_app

    yield c1_fastapi_app
    c1_fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def c1_client(c1_app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=c1_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def c2_app():
    from app.components.c2_case_intelligence.dev_app import app as c2_fastapi_app

    yield c2_fastapi_app


@pytest_asyncio.fixture
async def c2_client(c2_app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=c2_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_llm() -> Iterator[MagicMock]:
    """Controlled chat model for components that call get_llm()."""
    llm = MagicMock(name="MockChatModel")
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Mocked legal explanation."))
    llm.invoke = MagicMock(return_value=MagicMock(content="Mocked legal explanation."))
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value={"summary": "mocked"})
    llm.with_structured_output = MagicMock(return_value=structured)
    with patch("app.core.llm.get_llm", return_value=llm):
        yield llm


@pytest.fixture
def assert_no_production_database() -> None:
    """Fail fast if a Neon / production-looking URL leaked into the test process."""
    from app.core.config import settings

    for value in (settings.database_url, settings.database_url_direct):
        if not value:
            continue
        lowered = value.lower()
        if "neon.tech" in lowered or "production" in lowered:
            pytest.fail(f"Refusing to run tests against production-like database URL: {value!r}")
