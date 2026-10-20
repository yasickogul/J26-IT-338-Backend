"""Rule-based legal information extraction for Stage 2 (no Stage 3 retrieval).

Provenance rules:
- Prefer explicit intake fields (user-provided).
- Narrative CASE_NO / judgment cites → cited_case_references / LEGAL_REF entities,
  never current case_information.case_number unless the user supplied case_number.
- Do not invent party positions, evidence items, or legal provisions.
"""

from __future__ import annotations

import re

from app.components.c1_CaseAnalysis.config import (
    DISCLAIMER,
    DISPUTE_CATEGORIES,
    MODEL_VERSION,
)
from app.components.c1_CaseAnalysis.schemas import (
    CaseInformation,
    CaseIntakeRequest,
    ExtractedLegalInfo,
    LegalIssueItem,
    PartyPosition,
)
from app.shared.contracts.legal_info import LegalEntity, StructuredLegalInfo

_CASE_NO_RE = re.compile(
    r"\b(?:CA|SC|HC|DC)/(?:[A-Z]{1,6}/)?\d{1,4}/\d{2,4}\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_VS_RE = re.compile(
    r"(?P<p>.+?)\s+(?:v(?:s)?\.?|versus)\s+(?P<d>.+?)(?:\.|$|\n)",
    re.IGNORECASE | re.DOTALL,
)
_COURT_RE = re.compile(
    r"\b(Supreme Court|Court of Appeal|High Court|District Court|"
    r"Magistrate(?:'s)? Court|Commercial High Court)(?:\s+of\s+[\w\s]+)?\b",
    re.IGNORECASE,
)
def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def _parties_from_intake(intake: CaseIntakeRequest) -> list[PartyPosition]:
    """Only names/roles from intake — never invent a party position narrative."""
    parties: list[PartyPosition] = []
    if intake.plaintiff:
        parties.append(
            PartyPosition(party=intake.plaintiff.strip(), role="plaintiff", position=None)
        )
    if intake.defendant:
        parties.append(
            PartyPosition(party=intake.defendant.strip(), role="defendant", position=None)
        )
    return parties


def _parties_from_text(text: str) -> list[PartyPosition]:
    """Fallback when intake omits parties: extract names from an explicit 'A v. B' pattern."""
    # Prefer the parties block if present
    for chunk in (text or "").split("\n\n"):
        if chunk.lower().startswith("parties:"):
            m = _VS_RE.search(chunk)
            if m:
                break
    else:
        m = _VS_RE.search(text[:2000] if text else "")
    if not m:
        return []
    plaintiff = _norm(m.group("p"))
    defendant = _norm(m.group("d"))
    # Strip leading labels like "Parties:"
    if plaintiff and plaintiff.lower().startswith("parties:"):
        plaintiff = _norm(plaintiff.split(":", 1)[-1])
    out: list[PartyPosition] = []
    if plaintiff:
        out.append(PartyPosition(party=plaintiff[:240], role="plaintiff", position=None))
    if defendant:
        out.append(PartyPosition(party=defendant[:240], role="defendant", position=None))
    return out


def _collect_case_references(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _CASE_NO_RE.finditer(text or "")]


def _entities_from_text(
    text: str,
    *,
    current_case_number: str | None,
) -> tuple[list[LegalEntity], list[str]]:
    """Return (entities, cited_case_references).

    Narrative case numbers become LEGAL_REF (cited references), not the current case_number.
    """
    entities: list[LegalEntity] = []
    cited: list[str] = []
    current_norm = (current_case_number or "").strip().upper()

    for raw, start, end in _collect_case_references(text):
        if current_norm and raw.upper() == current_norm:
            entities.append(LegalEntity(text=raw, label="CASE_NO", start=start, end=end))
            continue
        # Any other CASE_NO pattern in the narrative is treated as a cited reference.
        entities.append(LegalEntity(text=raw, label="LEGAL_REF", start=start, end=end))
        if raw not in cited:
            cited.append(raw)

    for m in _DATE_RE.finditer(text or ""):
        entities.append(LegalEntity(text=m.group(0), label="DATE", start=m.start(), end=m.end()))
    for m in _COURT_RE.finditer(text or ""):
        entities.append(LegalEntity(text=m.group(0), label="COURT", start=m.start(), end=m.end()))
    return entities, cited


def _claims_from_intake(intake: CaseIntakeRequest) -> list[str]:
    """Claims from legal_issue + optional claim-like lines in `text` only.

    The case description is the narrative/summary source — it is not copied wholesale
    into claims even if it begins with phrasing like "The plaintiff claims…".
    """
    claims: list[str] = []
    if intake.legal_issue:
        claims.append(intake.legal_issue.strip())

    if intake.text:
        for line in intake.text.splitlines():
            low = line.lower().strip()
            if low.startswith(
                ("claim", "prayed", "seeks", "seeking", "alleges", "the plaintiff claims")
            ):
                cleaned = line.strip()
                # Prefer the first sentence of a long claim line.
                if "." in cleaned:
                    cleaned = cleaned.split(".", 1)[0].strip() + "."
                if cleaned and cleaned not in claims:
                    claims.append(cleaned[:400])
            if len(claims) >= 8:
                break
    return claims


def _evidence_list(intake: CaseIntakeRequest) -> list[str]:
    """Only user-supplied evidence factors — do not invent from keyword heuristics."""
    return [e.strip() for e in intake.evidence_factors if e and e.strip()]


def extract_legal_info(
    *,
    intake_id: str,
    intake: CaseIntakeRequest,
    processed_text: str,
) -> ExtractedLegalInfo:
    text = processed_text or ""
    category = intake.dispute_category
    case_type = DISPUTE_CATEGORIES.get(category, category)

    # Current case number: only from explicit intake (never from narrative citations).
    current_case_number = _norm(intake.case_number)

    parties = _parties_from_intake(intake)
    if not parties:
        parties = _parties_from_text(text)

    entities, cited_refs = _entities_from_text(text, current_case_number=current_case_number)
    # Court for the current case: intake only (narrative courts may belong to cited judgments).
    court = _norm(intake.court_type)
    claims = _claims_from_intake(intake)
    evidence = _evidence_list(intake)

    issues: list[LegalIssueItem] = []
    if intake.legal_issue:
        issues.append(
            LegalIssueItem(
                issue=intake.legal_issue.strip(),
                description=_norm(intake.description),
            )
        )

    # Summary is the user's description only — do not synthesize from merged boilerplate.
    summary = _norm(intake.description)

    title = _norm(intake.title)
    if not title and parties:
        names = [p.party for p in parties[:2]]
        if len(names) == 2:
            title = f"{names[0]} v. {names[1]}"
        elif names:
            title = names[0]
    # If still missing, leave null (do not invent a category-based title).

    case_information = CaseInformation(
        case_type=case_type,
        dispute_category=category,
        court=court,
        case_number=current_case_number,
        title=title,
        summary=summary,
        remedy_requested=_norm(intake.remedy),
        province=_norm(intake.province),
        district=_norm(intake.district),
        amount_lkr=_norm(intake.amount_lkr),
        relevant_dates=_norm(intake.relevant_dates),
    )

    structured = StructuredLegalInfo(
        document_id=intake_id,
        document_type="petition",
        case_type=case_type,
        legal_issue=issues[0].issue if issues else None,
        claims=claims,
        remedy_requested=case_information.remedy_requested,
        parties=[p.party for p in parties],
        evidence=evidence,
        court=court,
        entities=entities,
        raw_text_ref=f"intake:{intake_id}",
        confidence=0.72 if (parties or claims or evidence or summary) else 0.4,
    )

    return ExtractedLegalInfo(
        intake_id=intake_id,
        status="done",
        case_information=case_information,
        parties_and_positions=parties,
        legal_issues=issues,
        claims=claims,
        evidence_mentioned=evidence,
        cited_case_references=cited_refs,
        entities=[e.model_dump() for e in entities],
        structured_legal_info=structured,
        source_text_chars=len(text),
        model_version=MODEL_VERSION,
        disclaimer=DISCLAIMER,
    )


def intake_to_source_text(intake: CaseIntakeRequest, document_text: str = "") -> str:
    """Merge structured intake fields + optional document text into one narrative."""
    blocks: list[str] = []
    from app.components.c1_CaseAnalysis.config import DISPUTE_CATEGORY_LABELS

    label = DISPUTE_CATEGORY_LABELS.get(intake.dispute_category, intake.dispute_category)
    blocks.append(f"Dispute category: {label}")
    if intake.title:
        blocks.append(f"Case title: {intake.title}")
    if intake.case_number:
        blocks.append(f"Case number: {intake.case_number}")
    if intake.plaintiff or intake.defendant:
        blocks.append(
            f"Parties: {intake.plaintiff or 'N/A'} v. {intake.defendant or 'N/A'}"
        )
    if intake.court_type:
        blocks.append(f"Court: {intake.court_type}")
    if intake.province or intake.district:
        blocks.append(f"Jurisdiction: {intake.province or ''} / {intake.district or ''}".strip(" /"))
    if intake.amount_lkr:
        blocks.append(f"Dispute amount (LKR): {intake.amount_lkr}")
    if intake.relevant_dates:
        blocks.append(f"Relevant dates: {intake.relevant_dates}")
    if intake.legal_issue:
        blocks.append(f"Legal issue: {intake.legal_issue}")
    if intake.remedy:
        blocks.append(f"Remedy requested: {intake.remedy}")
    if intake.description:
        blocks.append(f"Case description:\n{intake.description}")
    if intake.evidence_factors:
        blocks.append("Evidence factors:\n- " + "\n- ".join(intake.evidence_factors))
    if intake.text:
        blocks.append(intake.text)
    if document_text:
        blocks.append(document_text)
    return "\n\n".join(blocks)
