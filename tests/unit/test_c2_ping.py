"""C2 skeleton endpoints currently implemented (ping only)."""

from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_c2_ping_contract(c2_client):
    response = await c2_client.get("/api/v1/c2/ping")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "c2"
    assert body["status"] == "ok"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_c2_unknown_path_returns_404(c2_client):
    """Civil intake lives under C1 today; C2 has no intake route yet."""
    response = await c2_client.post("/api/v1/c2/intake", json={"dispute_category": "employment"})
    assert response.status_code == 404
