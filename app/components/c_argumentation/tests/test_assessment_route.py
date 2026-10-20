from datetime import date
from uuid import uuid4

from fastapi.testclient import TestClient

from app.components.c_argumentation import router
from app.components.c_argumentation.dev_app import app
from app.components.c_argumentation.schemas_multi_agent import MultiAgentAssessment


def test_assessment_route_returns_c4_assessment(monkeypatch):
    expected = MultiAgentAssessment(
        case_summary="Property dispute",
        stated_facts="A deed is disputed.",
        plaintiff_arguments={"arguments": []},
        defense_counterarguments={"counterarguments": []},
        evidence_sufficiency="insufficient",
        neutral_case_assessment={"evidence_sufficiency": "insufficient", "assessment": "Insufficient."},
    )

    async def fake_assess(case):
        assert case.case_title == "Property dispute"
        return expected

    monkeypatch.setattr(router, "assess_case", fake_assess)
    response = TestClient(app).post(
        "/api/v1/c_argumentation/assess",
        json={
            "case_title": "Property dispute",
            "dispute_type": "Property ownership",
            "case_facts": "A deed is disputed and possession is contested by both parties.",
            "user_position": "The user claims ownership.",
            "opposing_position": "The other party disputes the deed.",
        },
    )

    assert response.status_code == 200
    assert response.json()["evidence_sufficiency"] == "insufficient"
