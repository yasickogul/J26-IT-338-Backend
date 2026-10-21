from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExtractedSectionResponse(BaseModel):
    label: str
    text: str = Field(max_length=8_000)


class DocumentExtractionResponse(BaseModel):
    document_id: str
    sanitized_filename: str
    file_type: Literal["pdf", "docx"]
    file_size: int = Field(ge=1)
    extraction_status: Literal["extracted", "ocr_required", "failed"]
    page_count: int | None = Field(default=None, ge=0)
    sections: list[ExtractedSectionResponse]
    extracted_text: str
    character_count: int = Field(ge=0)
    warnings: list[str]
    ocr_required: bool
    content_status: Literal["unverified_extracted_text"] = "unverified_extracted_text"


class DocumentEvidenceInput(BaseModel):
    """Client-provided extraction result, revalidated before agent use."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=100)
    sanitized_filename: str = Field(min_length=1, max_length=255)
    file_type: Literal["pdf", "docx"]
    file_size: int = Field(ge=1, le=10 * 1024 * 1024)
    extraction_status: Literal["extracted", "ocr_required", "failed"]
    page_count: int | None = Field(default=None, ge=0, le=500)
    sections: list[ExtractedSectionResponse] = Field(default_factory=list, max_length=40)
    extracted_text: str = Field(default="", max_length=8_000)
    character_count: int = Field(ge=0, le=8_000)
    warnings: list[str] = Field(default_factory=list, max_length=20)
    ocr_required: bool = False
    content_status: Literal["unverified_extracted_text"]

    @field_validator("extraction_status")
    @classmethod
    def reject_failed_documents(cls, value: str) -> str:
        if value == "failed":
            raise ValueError("failed documents cannot be included in an assessment")
        return value

    @model_validator(mode="after")
    def enforce_section_budget(self):
        if sum(len(section.text) for section in self.sections) > 8_000:
            raise ValueError("document section text exceeds the 8,000 character limit")
        return self
