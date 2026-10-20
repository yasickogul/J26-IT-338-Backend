"""Request/response models for C1 Stage 2 — case intake & legal information extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.shared.contracts.legal_info import StructuredLegalInfo

DisputeCategoryId = Literal[
    "contract",
    "property_land",
    "employment",
    "rental",
    "other_civil",
]

IntakeStatus = Literal["queued", "processing", "done", "failed"]

EvidenceAvailability = Literal[
    "available",
    "not_available",
    "not_sure",
    "not_applicable",
    "not_provided",
]


class EvidenceFactorItem(BaseModel):
    """Structured evidence row from the Lawyer wizard (Stage 2 intake only)."""

    id: str | None = None
    label: str
    availability: EvidenceAvailability = "not_provided"
    description: str | None = None


class CaseIntakeRequest(BaseModel):
    """Structured current-case intake from the frontend wizard (or API clients)."""

    dispute_category: DisputeCategoryId
    title: str | None = None
    case_number: str | None = None  # current case only; not inferred from narrative cites
    plaintiff: str | None = None
    defendant: str | None = None
    court_type: str | None = None
    province: str | None = None
    district: str | None = None
    amount_lkr: str | None = None
    relevant_dates: str | None = None
    case_stage: str | None = None
    case_stage_label: str | None = None
    subcategory: str | None = None
    subcategory_label: str | None = None
    other_category_description: str | None = None
    description: str | None = None
    legal_issue: str | None = None
    remedy: str | None = None
    evidence_factors: list[str] = Field(default_factory=list)
    evidence_items: list[EvidenceFactorItem] = Field(default_factory=list)
    text: str | None = None  # optional pasted case narrative / document text

    @model_validator(mode="after")
    def _derive_evidence_factors(self) -> CaseIntakeRequest:
        """Keep legacy `evidence_factors` populated when only structured items are sent."""
        if self.evidence_factors or not self.evidence_items:
            return self
        factors: list[str] = []
        for item in self.evidence_items:
            label = (item.label or "").strip()
            if not label:
                continue
            availability = item.availability.replace("_", " ")
            if item.availability == "available" and item.description:
                factors.append(f"{label} — available: {item.description.strip()}")
            elif item.availability == "not_provided":
                continue
            else:
                factors.append(f"{label} — {availability}")
        self.evidence_factors = factors
        return self


class ExtractTextRequest(BaseModel):
    text: str = Field(min_length=1)
    dispute_category: DisputeCategoryId | None = None


class PartyPosition(BaseModel):
    party: str
    role: Literal["plaintiff", "defendant", "appellant", "respondent", "other"] = "other"
    position: str | None = None


class LegalIssueItem(BaseModel):
    issue: str
    description: str | None = None


class CaseInformation(BaseModel):
    case_type: str | None = None  # e.g. Employment (intake label, NOT corpus "civil")
    dispute_category: DisputeCategoryId | None = None
    court: str | None = None
    case_number: str | None = None
    title: str | None = None
    summary: str | None = None
    remedy_requested: str | None = None
    province: str | None = None
    district: str | None = None
    amount_lkr: str | None = None
    relevant_dates: str | None = None
    case_stage: str | None = None
    case_stage_label: str | None = None
    subcategory: str | None = None
    subcategory_label: str | None = None
    other_category_description: str | None = None


class ExtractedLegalInfo(BaseModel):
    """Stage 2 output only — no similar cases / laws / report assembly (Stage 3+)."""

    intake_id: str
    status: IntakeStatus = "done"
    case_information: CaseInformation
    parties_and_positions: list[PartyPosition] = Field(default_factory=list)
    legal_issues: list[LegalIssueItem] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    evidence_mentioned: list[str] = Field(default_factory=list)
    cited_case_references: list[str] = Field(
        default_factory=list,
        description="Historical judgment / legal references found in the narrative (not the current case_number)",
    )
    entities: list[dict[str, Any]] = Field(default_factory=list)
    structured_legal_info: StructuredLegalInfo
    source_text_chars: int = 0
    model_version: str
    disclaimer: str
    created_at: datetime | None = None
    # Explicit Stage 3+ capability flags so clients never assume retrieval/scoring.
    similar_cases_available: bool = False
    legal_relevance_score_available: bool = False
    llm_explanation_available: bool = False


class IntakeAccepted(BaseModel):
    intake_id: str
    status: IntakeStatus
    message: str = "Case intake accepted"


class IntakeStatusResponse(BaseModel):
    intake_id: str
    status: IntakeStatus
    result: ExtractedLegalInfo | None = None
    error: str | None = None


# --- Stage 3: semantic retrieval + grounded analysis ---

RETRIEVAL_SCORE_EXPLANATION = (
    "retrieval_similarity is cosine similarity between the query embedding and the best "
    "matching judgment chunk. It is not a Legal Precedent Relevance Score and not a "
    "prediction of litigation success."
)
DISCLAIMER_TEXT = "This is informational and not legal advice."


class SimilarCasesRequest(BaseModel):
    query: str = Field(min_length=20, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filter_civil: bool = True


class SimilarJudgmentMatch(BaseModel):
    document_id: str
    title: str
    citation: str | None = None
    court: str | None = None
    decision_date: str | None = None
    source_url: str | None = None
    source_type: str = "judgment"
    case_type: str | None = None
    chunk_id: str
    chunk_index: int
    section: str | None = None
    supporting_excerpt: str
    retrieval_similarity: float = Field(
        description="Cosine similarity from pgvector (1 - distance). Not a legal success probability.",
    )
    score_label: str = "retrieval_similarity_cosine"


class SimilarCasesResponse(BaseModel):
    query: str
    retrieved_count: int
    cases: list[SimilarJudgmentMatch] = Field(default_factory=list)
    retrieval_method: str = "semantic_pgvector_cosine"
    # Literal defaults avoid NameError if this module is partially reloaded.
    score_explanation: str = (
        "retrieval_similarity is cosine similarity between the query embedding and the best "
        "matching judgment chunk. It is not a Legal Precedent Relevance Score and not a "
        "prediction of litigation success."
    )
    disclaimer: str = "This is informational and not legal advice."
    intake_id: str | None = None


class IntakeSimilarCasesRequest(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=20)
    filter_civil: bool = True
    include_analysis: bool = True
    use_llm: bool = False


class CaseAnalysisSection(BaseModel):
    title: str
    content: str
    source_document_ids: list[str] = Field(default_factory=list)


class GroundedCaseAnalysis(BaseModel):
    case_summary: str
    similar_historical_cases: list[CaseAnalysisSection] = Field(default_factory=list)
    legal_issues: list[str] = Field(default_factory=list)
    applicable_legal_principles: list[CaseAnalysisSection] = Field(default_factory=list)
    arguments_for_applicant: list[str] = Field(default_factory=list)
    arguments_for_respondent: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    llm_used: bool = False
    llm_status: str = "not_requested"
    model_version: str


class IntakeSimilarCasesResponse(BaseModel):
    intake_id: str
    query: str
    retrieved_count: int
    cases: list[SimilarJudgmentMatch] = Field(default_factory=list)
    analysis: GroundedCaseAnalysis | None = None
    retrieval_method: str = "semantic_pgvector_cosine"
    score_explanation: str = (
        "retrieval_similarity is cosine similarity between the query embedding and the best "
        "matching judgment chunk. It is not a Legal Precedent Relevance Score and not a "
        "prediction of litigation success."
    )
    disclaimer: str = "This is informational and not legal advice."
