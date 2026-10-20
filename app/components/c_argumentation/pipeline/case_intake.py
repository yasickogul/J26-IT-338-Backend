
"""Normalize structured or free-text case facts for the debate pipeline."""

from collections.abc import Mapping

from ..schemas import CaseInput, CaseResponse


def parse_case_facts(case_facts: Mapping[str, object] | str) -> dict[str, object]:
    """Return a stable case-facts mapping without modifying caller data."""
    if isinstance(case_facts, str):
        return {"facts": case_facts.strip()}
    return dict(case_facts)


def process_case(case: CaseInput) -> CaseResponse:
    """Validate and normalize a submitted civil dispute."""

    normalized_facts = parse_case_facts(case.case_facts)

    return CaseResponse(
        case_title=case.case_title.strip(),
        dispute_type=case.dispute_type.strip(),
        case_facts=str(normalized_facts["facts"]),
        user_position=case.user_position.strip(),
        opposing_position=(
            case.opposing_position.strip()
            if case.opposing_position
            else None
        ),
        status="case_received",
    )

def prepare_case_for_debate(case: CaseInput) -> dict[str, object]:
    """Prepare validated case information for the debate pipeline."""

    normalized = process_case(case)

    return {
        "case_title": normalized.case_title,
        "dispute_type": normalized.dispute_type,
        "facts": normalized.case_facts,
        "user_position": normalized.user_position,
        "opposing_position": normalized.opposing_position,
        "legal_issues": [],
        "available_evidence": [],
        "potential_evidence_gaps": [],
        "status": "ready_for_analysis",
    }
