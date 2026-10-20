"""Build a concise retrieval query from intake / extraction fields only."""

from __future__ import annotations

from typing import Any

from app.components.c1_CaseAnalysis.config import DISPUTE_CATEGORY_LABELS
from app.components.c1_CaseAnalysis.schemas import CaseIntakeRequest, ExtractedLegalInfo


def _clean(value: str | None, *, max_len: int = 600) -> str | None:
    if not value:
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def build_retrieval_query(
    *,
    description: str | None = None,
    legal_issue: str | None = None,
    remedy: str | None = None,
    dispute_category: str | None = None,
    subcategory_label: str | None = None,
    claims: list[str] | None = None,
    evidence_factors: list[str] | None = None,
    title: str | None = None,
) -> str:
    """Compose a search query from user-supplied / extracted facts only.

    Does not invent facts and does not use historical citations as shortcuts.
    """
    parts: list[str] = []
    if dispute_category:
        label = DISPUTE_CATEGORY_LABELS.get(dispute_category, dispute_category)
        parts.append(f"Dispute category: {label}")
    if subcategory_label:
        parts.append(f"Subcategory: {subcategory_label}")
    if title:
        cleaned_title = _clean(title, max_len=160)
        if cleaned_title:
            parts.append(f"Case title: {cleaned_title}")
    desc = _clean(description, max_len=1200)
    if desc:
        parts.append(desc)
    issue = _clean(legal_issue, max_len=400)
    if issue:
        parts.append(f"Legal issue: {issue}")
    rem = _clean(remedy, max_len=300)
    if rem:
        parts.append(f"Remedy sought: {rem}")
    for claim in (claims or [])[:4]:
        cleaned = _clean(claim, max_len=240)
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    for factor in (evidence_factors or [])[:6]:
        cleaned = _clean(factor, max_len=160)
        if cleaned:
            parts.append(f"Evidence: {cleaned}")

    query = "\n".join(parts).strip()
    if not query:
        raise ValueError("RETRIEVAL_QUERY_EMPTY: No case narrative or legal facts available to search.")
    return query


def build_retrieval_query_from_intake(intake: CaseIntakeRequest) -> str:
    return build_retrieval_query(
        description=intake.description or intake.text,
        legal_issue=intake.legal_issue,
        remedy=intake.remedy,
        dispute_category=intake.dispute_category,
        subcategory_label=intake.subcategory_label or intake.subcategory,
        evidence_factors=list(intake.evidence_factors or []),
        title=intake.title,
    )


def build_retrieval_query_from_extraction(
    extracted: ExtractedLegalInfo,
    *,
    intake_json: dict[str, Any] | None = None,
) -> str:
    info = extracted.case_information
    intake = intake_json or {}
    claims = list(extracted.claims or [])
    if extracted.legal_issues:
        claims = [item.issue for item in extracted.legal_issues] + claims
    return build_retrieval_query(
        description=info.summary or intake.get("description") or intake.get("text"),
        legal_issue=(
            extracted.legal_issues[0].issue
            if extracted.legal_issues
            else intake.get("legal_issue")
        ),
        remedy=info.remedy_requested or intake.get("remedy"),
        dispute_category=info.dispute_category or intake.get("dispute_category"),
        subcategory_label=info.subcategory_label
        or info.subcategory
        or intake.get("subcategory_label")
        or intake.get("subcategory"),
        claims=claims,
        evidence_factors=list(extracted.evidence_mentioned or intake.get("evidence_factors") or []),
        title=info.title or intake.get("title"),
    )
