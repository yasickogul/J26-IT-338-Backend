"""Stage 2 — current case intake & legal information extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.components.c1_CaseAnalysis.dev_app import app
from app.components.c1_CaseAnalysis.pipeline.info_extractor import (
    extract_legal_info,
    intake_to_source_text,
)
from app.components.c1_CaseAnalysis.pipeline.preprocessor import clean_text
from app.components.c1_CaseAnalysis.schemas import CaseIntakeRequest

SAMPLE_INTAKE = CaseIntakeRequest(
    dispute_category="employment",
    title="Perera v. ABC Services",
    plaintiff="Kasun Perera",
    defendant="ABC Services (Pvt) Ltd",
    court_type="District Court — Colombo",
    province="Western",
    district="Colombo",
    amount_lkr="2500000",
    relevant_dates="2024-01-15",
    description=(
        "The employee alleges wrongful termination without statutory notice after "
        "raising concerns about unpaid overtime. CA/WRT/0201/2022 is cited as background."
    ),
    legal_issue="Wrongful termination and unpaid overtime",
    remedy="Reinstatement and compensation",
    evidence_factors=[
        "Written agreement available",
        "Payment records available",
        "Legal notice sent",
    ],
    text=(
        "Kasun Perera vs. ABC Services (Pvt) Ltd. "
        "The plaintiff claims the employer terminated employment without due process. "
        "Payment records and correspondence are available."
    ),
)


def test_clean_text_collapses_whitespace():
    assert clean_text("a \n\n\n b") == "a\n\nb"


def test_extract_legal_info_structured_fields():
    merged = intake_to_source_text(SAMPLE_INTAKE)
    result = extract_legal_info(
        intake_id="00000000-0000-0000-0000-000000000001",
        intake=SAMPLE_INTAKE,
        processed_text=merged,
    )
    assert result.status == "done"
    assert result.case_information.case_type == "Employment"
    assert result.case_information.dispute_category == "employment"
    assert result.case_information.remedy_requested == "Reinstatement and compensation"
    assert result.case_information.case_number is None  # not supplied on intake
    assert any(p.role == "plaintiff" and "Perera" in p.party for p in result.parties_and_positions)
    assert all(p.position is None for p in result.parties_and_positions)
    assert result.legal_issues and "Wrongful termination" in result.legal_issues[0].issue
    assert "Written agreement available" in result.evidence_mentioned
    assert "CA/WRT/0201/2022" in result.cited_case_references
    assert result.structured_legal_info.case_type == "Employment"
    assert result.similar_cases_available is False
    assert result.legal_relevance_score_available is False
    assert result.llm_explanation_available is False
    payload = result.model_dump()
    assert "similar_cases" not in payload or payload.get("similar_cases") in (None, [], False)
    assert "relevant_laws_and_precedents" not in payload


def test_structured_evidence_items_derive_factors_and_stage_fields():
    intake = CaseIntakeRequest(
        dispute_category="contract",
        title="Silva v. Fernando Traders",
        plaintiff="Nimal Silva",
        defendant="Fernando Traders",
        case_stage="not_filed",
        case_stage_label="Not yet filed",
        subcategory="non_payment",
        subcategory_label="Non-payment / debt recovery",
        description=(
            "A supplier delivered construction materials under a written agreement. "
            "The customer failed to pay the agreed amount despite several reminders."
        ),
        legal_issue="Breach of contract involving non-payment",
        remedy="Recovery of outstanding payment",
        evidence_items=[
            {
                "id": "written_agreement",
                "label": "Written agreement",
                "availability": "available",
                "description": "Signed supply agreement",
            },
            {
                "id": "invoices",
                "label": "Invoices",
                "availability": "available",
                "description": "Outstanding invoices",
            },
            {
                "id": "payment_receipts",
                "label": "Payment receipts",
                "availability": "not_available",
            },
        ],
    )
    assert any("Written agreement" in factor for factor in intake.evidence_factors)
    merged = intake_to_source_text(intake)
    assert "Case stage: Not yet filed" in merged
    assert "Dispute subcategory: Non-payment / debt recovery" in merged
    result = extract_legal_info(
        intake_id="00000000-0000-0000-0000-000000000077",
        intake=intake,
        processed_text=merged,
    )
    assert result.case_information.case_stage == "not_filed"
    assert result.case_information.subcategory_label == "Non-payment / debt recovery"
    assert any("Written agreement" in item for item in result.evidence_mentioned)


def test_historical_citation_not_used_as_current_case_number():
    """Regression: narrative judgment cites must not become case_information.case_number."""
    intake = CaseIntakeRequest(
        dispute_category="contract",
        title="Silva v. Fernando Traders",
        plaintiff="Nimal Silva",
        defendant="Fernando Traders",
        court_type="District Court — Gampaha",
        description=(
            "The plaintiff seeks unpaid invoice amounts under a supply agreement. "
            "Counsel cited CA/RII/0021/2024 and SC/APPEAL/12/2018 as background authority."
        ),
        legal_issue="Breach of contract — unpaid invoices",
        remedy="Payment of LKR 1,200,000",
        evidence_factors=["Written agreement available", "Invoices available"],
    )
    merged = intake_to_source_text(intake)
    result = extract_legal_info(
        intake_id="00000000-0000-0000-0000-000000000099",
        intake=intake,
        processed_text=merged,
    )
    assert result.case_information.case_number is None
    assert "CA/RII/0021/2024" in result.cited_case_references
    assert any(e["label"] == "LEGAL_REF" and "CA/RII/0021/2024" in e["text"] for e in result.entities)
    assert not any(e["label"] == "CASE_NO" for e in result.entities)


def test_explicit_case_number_preserved_separately_from_cites():
    intake = CaseIntakeRequest(
        dispute_category="property_land",
        case_number="DC/COL/123/2024",
        plaintiff="A",
        defendant="B",
        description="Boundary dispute. Relied on CA/PHC/0004/2021.",
        legal_issue="Land ownership / boundary",
    )
    result = extract_legal_info(
        intake_id="00000000-0000-0000-0000-000000000088",
        intake=intake,
        processed_text=intake_to_source_text(intake),
    )
    assert result.case_information.case_number == "DC/COL/123/2024"
    assert "CA/PHC/0004/2021" in result.cited_case_references
    assert "CA/PHC/0004/2021" != result.case_information.case_number


def test_ping_reports_stage2():
    client = TestClient(app)
    r = client.get("/api/v1/c1/ping")
    assert r.status_code == 200
    assert r.json()["component"] == "c1"
    assert r.json()["stage"] == 2


def test_invalid_dispute_category_returns_422():
    client = TestClient(app)
    bad = SAMPLE_INTAKE.model_dump()
    bad["dispute_category"] = "criminal"
    r = client.post("/api/v1/c1/intake", json=bad)
    assert r.status_code == 422


def test_missing_required_category_returns_422():
    client = TestClient(app)
    r = client.post("/api/v1/c1/intake", json={"title": "No category"})
    assert r.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_corrupt_pdf_without_narrative_fails_safely(tmp_path: Path):
    from app.components.c1_CaseAnalysis.service import stage2_service
    from app.core.database import direct_engine

    junk = tmp_path / "not_a_pdf.pdf"
    junk.write_bytes(b"this is not a pdf file at all")
    intake = CaseIntakeRequest(dispute_category="employment")  # no narrative fallback
    try:
        with pytest.raises(ValueError, match="PDF_EXTRACTION_FAILED"):
            await stage2_service.process_intake(intake, pdf_path=junk, filename="not_a_pdf.pdf")
    except RuntimeError as exc:
        if "DATABASE_URL" in str(exc):
            pytest.skip("DATABASE_URL not configured")
        raise
    finally:
        if direct_engine is not None:
            await direct_engine.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_intake_persists_and_returns_json():
    from app.components.c1_CaseAnalysis.service import stage2_service
    from app.core.database import direct_engine

    try:
        result = await stage2_service.process_intake(SAMPLE_INTAKE)
    except RuntimeError as exc:
        if "DATABASE_URL" in str(exc):
            pytest.skip("DATABASE_URL not configured")
        raise

    assert result.status == "done"
    assert result.intake_id
    assert result.case_information.case_number is None
    assert "CA/WRT/0201/2022" in result.cited_case_references

    status = await stage2_service.get_intake_status(result.intake_id)
    assert status.status == "done"
    assert status.result is not None
    assert status.result.intake_id == result.intake_id
    assert status.result.case_information.case_number is None
    if direct_engine is not None:
        await direct_engine.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_intake_endpoint_json_roundtrip():
    from httpx import ASGITransport, AsyncClient

    from app.core.database import direct_engine

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/v1/c1/intake", json=SAMPLE_INTAKE.model_dump())
    if r.status_code == 500 and "DATABASE_URL" in r.text:
        pytest.skip("DATABASE_URL not configured")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "done"
    assert body["case_information"]["case_number"] is None
    assert "CA/WRT/0201/2022" in body["cited_case_references"]
    print("\n=== Stage 2 structured JSON ===")
    print(json.dumps(body, indent=2, default=str))
    if direct_engine is not None:
        await direct_engine.dispose()
