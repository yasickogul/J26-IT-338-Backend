from app.components.c_argumentation.agents.auditor_agent import audit_turn


def test_auditor_scores_valid_citation():
    assert audit_turn({"id": "n1"})["strength_score"] == 1.0
