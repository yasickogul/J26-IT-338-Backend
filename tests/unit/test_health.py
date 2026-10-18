"""System health endpoints — offline (no Neon)."""

from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_returns_ok_contract(client, assert_no_production_database):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_db_reports_error_when_database_disabled(client, assert_no_production_database):
    """With DATABASE_URL cleared for unit tests, /health/db must not pretend Neon is up."""
    response = await client.get("/health/db")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert "error" in body
