"""Evidence-grounded case analysis from intake facts + retrieved judgment excerpts.

Does not invent holdings, statutes, or success probabilities.
Optional LM Studio enrichment is applied only when explicitly requested and available.
"""

from __future__ import annotations

from typing import Any

from app.components.c1_CaseAnalysis.analysis.lm_studio import enrich_analysis_with_lm_studio
from app.components.c1_CaseAnalysis.config import ANALYSIS_MODEL_VERSION, RETRIEVAL_DISCLAIMER
from app.components.c1_CaseAnalysis.schemas import (
    CaseAnalysisSection,
    ExtractedLegalInfo,
    GroundedCaseAnalysis,
    SimilarJudgmentMatch,
)


def _truncate(text: str, max_len: int = 420) -> str:
    clean = " ".join(text.split()).strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "…"


def _case_summary(extracted: ExtractedLegalInfo, intake_json: dict[str, Any] | None) -> str:
    info = extracted.case_information
    bits: list[str] = []
    if info.title:
        bits.append(info.title)
    if info.case_type or info.dispute_category:
        bits.append(f"Category: {info.case_type or info.dispute_category}")
    parties = [p.party for p in extracted.parties_and_positions]
    if parties:
        bits.append("Parties: " + " v. ".join(parties[:2]))
    if info.summary:
        bits.append(info.summary)
    elif intake_json and intake_json.get("description"):
        bits.append(str(intake_json["description"]))
    if info.remedy_requested:
        bits.append(f"Remedy sought: {info.remedy_requested}")
    if not bits:
        return "Insufficient submitted facts were available to summarise the current case."
    return " ".join(_truncate(b, 500) for b in bits)


def _legal_issues(extracted: ExtractedLegalInfo, intake_json: dict[str, Any] | None) -> list[str]:
    issues = [item.issue for item in extracted.legal_issues if item.issue]
    for claim in extracted.claims:
        if claim and claim not in issues:
            issues.append(claim)
    if not issues and intake_json and intake_json.get("legal_issue"):
        issues.append(str(intake_json["legal_issue"]))
    if not issues:
        issues.append(
            "No explicit legal issue was supplied. Issues should be inferred only after reviewing "
            "the case narrative with counsel; this analysis does not invent legal questions."
        )
    return issues[:8]


def _arguments(extracted: ExtractedLegalInfo) -> tuple[list[str], list[str]]:
    applicant: list[str] = []
    respondent: list[str] = []
    info = extracted.case_information
    if info.summary:
        applicant.append(
            "Based on the submitted narrative (allegation, not an established finding): "
            + _truncate(info.summary, 280)
        )
    if extracted.legal_issues:
        applicant.append(
            "The applicant may frame the dispute around: "
            + "; ".join(item.issue for item in extracted.legal_issues[:3])
            + "."
        )
    if info.remedy_requested:
        applicant.append(f"The applicant seeks: {info.remedy_requested}.")
    if extracted.evidence_mentioned:
        applicant.append(
            "Evidence identified by the user as potentially available includes: "
            + "; ".join(extracted.evidence_mentioned[:4])
            + "."
        )
    else:
        applicant.append(
            "Limited evidence was listed in the intake; applicant arguments remain provisional."
        )

    respondent.append(
        "The respondent may dispute the factual narrative until tested through pleadings and evidence."
    )
    if info.remedy_requested:
        respondent.append(
            "The respondent may argue that the requested remedy is unavailable or disproportionate "
            "on the facts as currently stated."
        )
    if not extracted.evidence_mentioned:
        respondent.append(
            "Absence of listed contemporaneous documents may be used to challenge the strength of the claim."
        )
    else:
        respondent.append(
            "The respondent may contest authenticity, completeness, or legal effect of the listed evidence items."
        )

    if not applicant:
        applicant.append("Insufficient submitted facts to draft applicant-side arguments.")
    return applicant[:6], respondent[:6]


def _evidence_gaps(extracted: ExtractedLegalInfo, matches: list[SimilarJudgmentMatch]) -> list[str]:
    gaps: list[str] = []
    mentioned = {e.lower() for e in extracted.evidence_mentioned}
    if not extracted.evidence_mentioned:
        gaps.append("No evidence items were marked available in the intake.")
    if not any("agreement" in m or "contract" in m or "letter" in m for m in mentioned):
        gaps.append(
            "Consider whether primary written instruments (agreements, notices, board minutes, or orders) "
            "are available for review."
        )
    if not extracted.case_information.relevant_dates:
        gaps.append("Key chronology dates were not provided in the intake.")
    if not extracted.parties_and_positions:
        gaps.append("Party identities were incomplete in the structured extraction.")
    if matches:
        gaps.append(
            "Retrieved judgments may discuss additional documentary themes; compare those themes with "
            "the user's evidence list without treating retrieval similarity as proof of legal authority."
        )
    else:
        gaps.append(
            "No similar judgments were retrieved, so evidence-gap comparison against precedents is limited."
        )
    return gaps[:8]


def _similar_case_sections(matches: list[SimilarJudgmentMatch]) -> list[CaseAnalysisSection]:
    sections: list[CaseAnalysisSection] = []
    for match in matches:
        cite = match.citation or match.document_id
        why = (
            f"Retrieved as a semantically similar civil judgment (cosine similarity "
            f"{match.retrieval_similarity:.4f}). Supporting excerpt from chunk "
            f"{match.chunk_index}: {_truncate(match.supporting_excerpt, 360)}"
        )
        sections.append(
            CaseAnalysisSection(
                title=f"{cite} — {match.title}",
                content=why,
                source_document_ids=[match.document_id],
            )
        )
    return sections


def _principle_sections(matches: list[SimilarJudgmentMatch]) -> list[CaseAnalysisSection]:
    """Only surface principles that are literally supported by retrieved excerpts."""
    sections: list[CaseAnalysisSection] = []
    for match in matches[:5]:
        excerpt = _truncate(match.supporting_excerpt, 400)
        if len(excerpt) < 40:
            continue
        cite = match.citation or match.document_id
        sections.append(
            CaseAnalysisSection(
                title=f"Excerpt-supported theme from {cite}",
                content=(
                    "The following text is taken from the retrieved judgment chunk and is not a "
                    f"restated holding: \"{excerpt}\""
                ),
                source_document_ids=[match.document_id],
            )
        )
    if not sections:
        sections.append(
            CaseAnalysisSection(
                title="Insufficient retrieved excerpts",
                content=(
                    "Retrieved evidence was insufficient to identify applicable legal principles "
                    "without inventing holdings."
                ),
                source_document_ids=[],
            )
        )
    return sections


async def build_grounded_analysis(
    *,
    extracted: ExtractedLegalInfo,
    matches: list[SimilarJudgmentMatch],
    intake_json: dict[str, Any] | None = None,
    use_llm: bool = False,
) -> GroundedCaseAnalysis:
    applicant, respondent = _arguments(extracted)
    analysis = GroundedCaseAnalysis(
        case_summary=_case_summary(extracted, intake_json),
        similar_historical_cases=_similar_case_sections(matches),
        legal_issues=_legal_issues(extracted, intake_json),
        applicable_legal_principles=_principle_sections(matches),
        arguments_for_applicant=applicant,
        arguments_for_respondent=respondent,
        evidence_gaps=_evidence_gaps(extracted, matches),
        limitations=[
            RETRIEVAL_DISCLAIMER,
            "Semantic similarity does not establish binding effect, ratio decidendi, or likely outcome.",
            "Arguments below distinguish user allegations from adjudicated findings.",
            "No numerical success probability is provided.",
            "Every judgment-linked statement below is tied to a retrieved document ID or excerpt.",
        ],
        llm_used=False,
        llm_status="not_requested",
        model_version=ANALYSIS_MODEL_VERSION,
    )

    if use_llm:
        return await enrich_analysis_with_lm_studio(
            analysis=analysis,
            extracted=extracted,
            matches=matches,
        )
    return analysis
