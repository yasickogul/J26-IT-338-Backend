"""Root pytest bootstrap — runs before test collection imports application modules.

Default automated runs must never contact Neon, download embedding models, or call LLM APIs.
Set RUN_INTEGRATION_TESTS=1 to allow a real DATABASE_URL for explicit integration tests.
"""

from __future__ import annotations

import os

# Force offline/safe defaults before any `app.*` import.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LLM_MODEL", "mock-model-for-tests")
os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("EMBEDDING_DIM", "384")
os.environ.setdefault("FILE_STORAGE", "local")
os.environ.setdefault("FILE_STORAGE_PATH", "./data/uploads-test")

_ALLOW_INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS", "").strip() == "1"

if not _ALLOW_INTEGRATION:
    # Empty strings override .env values (pydantic-settings prefers real env vars).
    os.environ["DATABASE_URL"] = ""
    os.environ["DATABASE_URL_DIRECT"] = ""
    os.environ["LLM_API_KEY"] = ""


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: offline unit tests (default)")
    config.addinivalue_line(
        "markers",
        "integration: needs isolated Postgres/pgvector; excluded from default CI",
    )
    config.addinivalue_line(
        "markers",
        "live_llm: contacts LM Studio or a paid LLM API; opt-in only",
    )
