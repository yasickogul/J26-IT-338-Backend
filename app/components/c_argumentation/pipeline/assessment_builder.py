"""Build a compact assessment summary from an audit report."""

from collections.abc import Mapping


def build_assessment(audit_report: Mapping[str, object]) -> dict[str, object]:
    errors = list(audit_report.get("validation_errors", []))
    gaps = list(audit_report.get("evidence_gaps", []))
    return {
        "status": "needs_review" if errors or gaps else "ready",
        "validation_error_count": len(errors),
        "evidence_gap_count": len(gaps),
    }
