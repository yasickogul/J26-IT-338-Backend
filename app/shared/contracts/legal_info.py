from typing import Optional

from pydantic import BaseModel, Field


class LegalEntity(BaseModel):
    text: str
    label: str  # PERSON | ORG | COURT | LOCATION | DATE | CASE_NO | LEGAL_REF
    start: Optional[int] = None
    end: Optional[int] = None


class StructuredLegalInfo(BaseModel):
    """Output of C1; input to C2 / C3 / C4."""
    document_id: Optional[str] = None
    document_type: str = "other"  # judgment | agreement | petition | other
    case_type: Optional[str] = None
    legal_issue: Optional[str] = None
    claims: list[str] = Field(default_factory=list)
    remedy_requested: Optional[str] = None
    parties: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    court: Optional[str] = None
    entities: list[LegalEntity] = Field(default_factory=list)
    raw_text_ref: Optional[str] = None
    confidence: Optional[float] = None
