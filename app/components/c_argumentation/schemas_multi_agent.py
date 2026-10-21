"""Pydantic contracts for the C4 multi-agent case assessment prototype."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class EvidenceSource(BaseModel):
    source_id: str
    chunk_id: UUID
    judgment_id: UUID
    case_number: str
    start_page: int | None
    end_page: int | None
    pdf_url: str | None
    similarity_score: float | None
    chunk_text: str


class DocumentEvidenceSource(BaseModel):
    source_id: str
    document_id: str
    sanitized_filename: str
    section_label: str
    section_text: str
    extraction_status: Literal["extracted", "ocr_required"]
    warning: str | None = None


class EvidenceBundle(BaseModel):
    sources: list[EvidenceSource] = Field(default_factory=list)
    document_sources: list[DocumentEvidenceSource] = Field(default_factory=list)
    evidence_label: str
    evidence_explanation: str
    context_warnings: list[str] = Field(default_factory=list)


class PlaintiffArgument(BaseModel):
    claim: str
    supporting_principles: list[str] = Field(default_factory=list)
    evidence_source_ids: list[str] = Field(default_factory=list)
    evidence_support: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class PlaintiffOutput(BaseModel):
    role: Literal["plaintiff"] = "plaintiff"
    arguments: list[PlaintiffArgument] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unsupported_claim_warnings: list[str] = Field(default_factory=list)


class DefenseCounterargument(BaseModel):
    challenged_claim: str
    counterargument: str
    evidence_source_ids: list[str] = Field(default_factory=list)
    factual_disputes: list[str] = Field(default_factory=list)
    possible_defenses: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class DefenseOutput(BaseModel):
    role: Literal["defense"] = "defense"
    counterarguments: list[DefenseCounterargument] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unsupported_claim_warnings: list[str] = Field(default_factory=list)


class AuditorOutput(BaseModel):
    role: Literal["auditor"] = "auditor"
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    unsupported_claim_warnings: list[str] = Field(default_factory=list)
    contradictory_claims: list[str] = Field(default_factory=list)
    verified_source_ids: list[str] = Field(default_factory=list)
    evidence_sufficiency: Literal["sufficient", "limited", "insufficient"] = "insufficient"
    assessment: str = ""
    disclaimer: str = "Informational legal analysis only; not professional legal advice or a court verdict."


class CitationAudit(BaseModel):
    valid_source_ids: list[str] = Field(default_factory=list)
    invalid_source_ids: list[str] = Field(default_factory=list)
    invalid_case_numbers: list[str] = Field(default_factory=list)
    valid_document_source_ids: list[str] = Field(default_factory=list)
    invalid_page_references: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.invalid_source_ids and not self.invalid_case_numbers and not self.invalid_page_references


class MultiAgentAssessment(BaseModel):
    case_summary: str
    stated_facts: str
    plaintiff_arguments: PlaintiffOutput
    defense_counterarguments: DefenseOutput
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    verified_court_case_references: list[EvidenceSource] = Field(default_factory=list)
    uploaded_document_references: list[DocumentEvidenceSource] = Field(default_factory=list)
    document_context_warnings: list[str] = Field(default_factory=list)
    unsupported_claim_warnings: list[str] = Field(default_factory=list)
    evidence_sufficiency: Literal["sufficient", "limited", "insufficient"]
    neutral_case_assessment: AuditorOutput
    citation_audits: dict[str, CitationAudit] = Field(default_factory=dict)
    execution_errors: list[str] = Field(default_factory=list)
    disclaimer: str = "Informational legal analysis only; not professional legal advice or a court verdict."
