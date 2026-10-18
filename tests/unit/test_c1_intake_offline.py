"""C1 Stage 2 intake validation and mocked processing — no Neon."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.components.c1_CaseAnalysis.schemas import (
    CaseInformation,
    ExtractedLegalInfo,
)
from app.shared.contracts.legal_info import StructuredLegalInfo


def _fake_extraction(*, intake_id: str = "00000000-0000-0000-0000-000000000001") -> ExtractedLegalInfo:
    return ExtractedLegalInfo(
        intake_id=intake_id,
        status="done",
        case_information=CaseInformation(
            case_type="Employment",
            dispute_category="employment",
            summary="Synthetic termination without notice.",
            remedy_requested="Compensation",
        ),
        claims=["Wrongful termination"],
        evidence_mentioned=["Written agreement"],
        structured_legal_info=StructuredLegalInfo(
            document_id=intake_id,
            document_type="petition",
            case_type="Employment",
            legal_issue="Wrongful termination",
            claims=["Wrongful termination"],
            remedy_requested="Compensation",
            parties=["A", "B"],
            evidence=["Written agreement"],
            confidence=0.72,
        ),
        model_version="stage2-rules-v1",
        disclaimer="This is informational and not legal advice.",
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_c1_ping_reports_stage2(c1_client):
    response = await c1_client.get("/api/v1/c1/ping")
    assert response.status_code == 200
    body = response.json()
    assert body == {"component": "c1", "status": "ok", "stage": 2}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_c1_intake_rejects_invalid_dispute_category(c1_client):
    response = await c1_client.post(
        "/api/v1/c1/intake",
        json={"dispute_category": "criminal", "description": "Not a civil category."},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any("dispute_category" in str(item.get("loc", [])) for item in detail)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_c1_intake_rejects_missing_category(c1_client):
    response = await c1_client.post("/api/v1/c1/intake", json={"title": "No category"})
    assert response.status_code == 422


@pytest.mark.unit
@pytest.mark.asyncio
async def test_c1_intake_success_with_mocked_service(c1_client, monkeypatch):
    """Mock the Stage 2 service boundary so CI never opens Neon."""
    from app.components.c1_CaseAnalysis import router as c1_router

    fake = _fake_extraction()
    monkeypatch.setattr(
        c1_router.stage2_service,
        "process_intake",
        AsyncMock(return_value=fake),
    )

    response = await c1_client.post(
        "/api/v1/c1/intake",
        json={
            "dispute_category": "employment",
            "description": (
                "The employee was terminated without statutory notice after requesting unpaid overtime."
            ),
            "legal_issue": "Wrongful termination and unpaid wages",
            "remedy": "Compensation",
            "evidence_factors": ["Written agreement"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "done"
    assert body["intake_id"] == fake.intake_id
    assert body["case_information"]["dispute_category"] == "employment"
    assert body["case_information"]["summary"].startswith("Synthetic")
    assert "similar_cases" not in body
    assert body["model_version"] == "stage2-rules-v1"
