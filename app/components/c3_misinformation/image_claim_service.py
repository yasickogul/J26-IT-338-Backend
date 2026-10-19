"""OCR-to-claim extraction orchestration with no verification or persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .pipeline.claim_extraction import (
    ClaimExtractionProvider,
    ExtractionFailureReason,
    ExtractionStatus,
    extract_legal_claim,
)
from .pipeline.language_detection import (
    DetectionReason,
    DetectionStatus,
    LanguageLabel,
    detect_language,
)
from .pipeline.ocr_extractor import (
    ImageProcessingConfig,
    ImageProcessingError,
    ImageProcessor,
    OCRProvider,
    OCRProviderError,
    OCRProviderResult,
    PillowImageProcessor,
)
from .pipeline.text_preprocessing import clean_text

FailureStage = Literal[
    "image_validation", "image_processing", "ocr", "claim_extraction"
]
OCRStatus = Literal["completed", "failed"]


@dataclass(frozen=True)
class SafeExtractionError:
    code: str
    stage: FailureStage


@dataclass(frozen=True)
class ImageClaimExtractionResult:
    ocr_raw_text: str
    ocr_status: OCRStatus
    cleaned_text: str
    extracted_legal_claim: str | None
    ocr_confidence: float | None
    ocr_languages: tuple[str, ...]
    low_confidence: bool
    extraction_status: ExtractionStatus
    detected_language: LanguageLabel | None
    language_detection_status: DetectionStatus | Literal["not_run"]
    language_detection_reason: DetectionReason | None
    ocr_method: str | None = None
    claim_extraction_method: str | None = None
    error: SafeExtractionError | None = None


def _failed(
    *,
    languages: tuple[str, ...],
    error: SafeExtractionError,
    raw_text: str = "",
    cleaned_text: str = "",
    confidence: float | None = None,
    low_confidence: bool = True,
    ocr_method: str | None = None,
) -> ImageClaimExtractionResult:
    return ImageClaimExtractionResult(
        ocr_raw_text=raw_text,
        ocr_status="failed",
        cleaned_text=cleaned_text,
        extracted_legal_claim=None,
        ocr_confidence=confidence,
        ocr_languages=languages,
        low_confidence=low_confidence,
        extraction_status="failed",
        detected_language=None,
        language_detection_status="not_run",
        language_detection_reason=None,
        ocr_method=ocr_method,
        error=error,
    )


def extract_claim_from_image(
    image_bytes: bytes,
    *,
    ocr_provider: OCRProvider,
    claim_provider: ClaimExtractionProvider | None,
    image_config: ImageProcessingConfig,
    ocr_languages: tuple[str, ...] = ("eng", "sin", "tam"),
    low_confidence_threshold: float = 0.60,
    image_processor: ImageProcessor | None = None,
) -> ImageClaimExtractionResult:
    """Extract a legal assertion from an image without judging its truth."""
    if not isinstance(image_bytes, bytes):
        raise TypeError("image_bytes must be bytes")
    if not isinstance(ocr_languages, tuple) or not ocr_languages:
        raise ValueError("ocr_languages must be a non-empty tuple")
    if (
        isinstance(low_confidence_threshold, bool)
        or not isinstance(low_confidence_threshold, (int, float))
        or not 0 <= low_confidence_threshold <= 1
    ):
        raise ValueError("low_confidence_threshold must be between 0 and 1")

    processor = image_processor or PillowImageProcessor()
    try:
        processed = processor.preprocess(image_bytes, config=image_config)
    except ImageProcessingError as exc:
        stage: FailureStage = (
            "image_validation"
            if exc.reason
            in {
                "empty_image",
                "image_too_large",
                "unsupported_image_type",
                "corrupted_image",
                "decoded_image_too_large",
            }
            else "image_processing"
        )
        return _failed(
            languages=ocr_languages,
            error=SafeExtractionError(code=exc.reason, stage=stage),
        )

    try:
        ocr = ocr_provider.extract(processed, languages=ocr_languages)
    except OCRProviderError as exc:
        return _failed(
            languages=ocr_languages,
            error=SafeExtractionError(code=exc.reason, stage="ocr"),
        )
    except Exception:  # noqa: BLE001 - provider boundary must return a safe code
        return _failed(
            languages=ocr_languages,
            error=SafeExtractionError(code="ocr_provider_error", stage="ocr"),
        )

    if (
        not isinstance(ocr, OCRProviderResult)
        or not isinstance(ocr.method, str)
        or not ocr.method.strip()
    ):
        return _failed(
            languages=ocr_languages,
            error=SafeExtractionError(code="invalid_ocr_response", stage="ocr"),
        )
    confidence = ocr.confidence
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        return _failed(
            languages=ocr_languages,
            error=SafeExtractionError(code="invalid_ocr_response", stage="ocr"),
        )
    low_confidence = confidence is None or confidence < low_confidence_threshold
    if not isinstance(ocr.raw_text, str) or not ocr.raw_text.strip():
        return _failed(
            languages=ocr_languages,
            error=SafeExtractionError(code="empty_ocr_result", stage="ocr"),
            confidence=confidence,
            low_confidence=low_confidence,
            ocr_method=ocr.method.strip(),
        )

    raw_text = ocr.raw_text
    cleaned = clean_text(raw_text)
    if not cleaned:
        return _failed(
            languages=ocr_languages,
            error=SafeExtractionError(code="empty_ocr_result", stage="ocr"),
            raw_text=raw_text,
            confidence=confidence,
            low_confidence=low_confidence,
            ocr_method=ocr.method.strip(),
        )

    language_detection = detect_language(cleaned)
    extraction = extract_legal_claim(cleaned, provider=claim_provider)
    error = None
    if extraction.status == "failed":
        reason: ExtractionFailureReason = extraction.failure_reason or "provider_error"
        error = SafeExtractionError(
            code=f"claim_extraction_{reason}",
            stage="claim_extraction",
        )
    return ImageClaimExtractionResult(
        ocr_raw_text=raw_text,
        ocr_status="completed",
        cleaned_text=cleaned,
        extracted_legal_claim=extraction.extracted_claim,
        ocr_confidence=confidence,
        ocr_languages=ocr_languages,
        low_confidence=low_confidence,
        extraction_status=extraction.status,
        detected_language=language_detection.label,
        language_detection_status=language_detection.status,
        language_detection_reason=language_detection.reason,
        ocr_method=ocr.method.strip(),
        claim_extraction_method=extraction.method,
        error=error,
    )
