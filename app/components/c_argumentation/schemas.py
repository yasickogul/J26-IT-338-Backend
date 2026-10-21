
from pydantic import BaseModel, Field
from .schemas_document import DocumentEvidenceInput


class CaseInput(BaseModel):
    case_title: str = Field(min_length=3)
    dispute_type: str = Field(min_length=3)
    case_facts: str = Field(min_length=20)
    user_position: str = Field(min_length=3)
    opposing_position: str | None = None
    documents: list[DocumentEvidenceInput] = Field(default_factory=list, max_length=5)


class CaseResponse(BaseModel):
    case_title: str
    dispute_type: str
    case_facts: str
    user_position: str
    opposing_position: str | None = None
    status: str
