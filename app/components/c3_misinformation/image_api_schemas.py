"""Schemas for image-based legal claim extraction."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ImageExtractionErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    stage: Literal["image_validation", "image_processing", "ocr", "claim_extraction"]


class ImageClaimExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ocr_raw_text: str
    ocr_status: Literal["completed", "failed"]
    cleaned_text: str
    extracted_legal_claim: str | None
    ocr_confidence: float | None
    ocr_languages: list[str]
    low_confidence: bool
    extraction_status: Literal[
        "extracted",
        "multiple_claims",
        "ambiguous",
        "no_legal_assertion",
        "failed",
    ]
    detected_language: Literal["en", "si", "ta", "mixed"] | None
    language_detection_status: Literal[
        "detected",
        "uncertain",
        "unsupported",
        "invalid",
        "detector_error",
        "not_run",
    ]
    language_detection_reason: str | None
    ocr_method: str | None = None
    claim_extraction_method: str | None = None
    error: ImageExtractionErrorResponse | None = None
