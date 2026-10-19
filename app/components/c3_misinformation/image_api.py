"""C3-owned multipart API for OCR claim extraction only."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, File, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from .image_api_schemas import ImageClaimExtractionResponse
from .image_claim_service import (
    ImageClaimExtractionResult,
    SafeExtractionError,
    extract_claim_from_image,
)
from .pipeline.claim_extraction import GeminiClaimExtractionProvider
from .pipeline.ocr_extractor import ImageProcessingConfig, TesseractOCRProvider

router = APIRouter(tags=["C3 image claim extraction"])
ImageClaimExtractor = Callable[[bytes], ImageClaimExtractionResult]
_SUPPORTED_IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png"}

_DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024
_DEFAULT_MAX_DECODED_PIXELS = 40_000_000


def _positive_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def get_max_image_bytes() -> int:
    return _positive_env_int("C3_OCR_MAX_IMAGE_BYTES", _DEFAULT_MAX_FILE_BYTES)


def _confidence_threshold() -> float:
    value = float(os.getenv("C3_OCR_LOW_CONFIDENCE_THRESHOLD", "0.60"))
    if not 0 <= value <= 1:
        raise ValueError("C3_OCR_LOW_CONFIDENCE_THRESHOLD must be between 0 and 1")
    return value


def _ocr_languages() -> tuple[str, ...]:
    languages = tuple(
        item.strip()
        for item in os.getenv("C3_OCR_LANGUAGES", "eng,sin,tam").split(",")
        if item.strip()
    )
    if not languages:
        raise ValueError("C3_OCR_LANGUAGES must contain at least one language")
    return languages


@lru_cache(maxsize=1)
def get_image_claim_extractor() -> ImageClaimExtractor:
    config = ImageProcessingConfig(
        max_file_bytes=get_max_image_bytes(),
        max_decoded_pixels=_positive_env_int(
            "C3_OCR_MAX_DECODED_PIXELS",
            _DEFAULT_MAX_DECODED_PIXELS,
        ),
        improve_contrast=os.getenv("C3_OCR_IMPROVE_CONTRAST", "true").casefold()
        in {"1", "true", "yes", "on"},
        reduce_noise=os.getenv("C3_OCR_REDUCE_NOISE", "false").casefold()
        in {"1", "true", "yes", "on"},
    )
    ocr_provider = TesseractOCRProvider()
    claim_provider = GeminiClaimExtractionProvider()
    languages = _ocr_languages()
    threshold = _confidence_threshold()

    def extractor(image_bytes: bytes) -> ImageClaimExtractionResult:
        return extract_claim_from_image(
            image_bytes,
            ocr_provider=ocr_provider,
            claim_provider=claim_provider,
            image_config=config,
            ocr_languages=languages,
            low_confidence_threshold=threshold,
        )

    return extractor


@router.post(
    "/claims/extract-image",
    response_model=ImageClaimExtractionResponse,
)
async def extract_image_claim(
    image: Annotated[UploadFile, File()],
    response: Response,
    extractor: Annotated[
        ImageClaimExtractor,
        Depends(get_image_claim_extractor),
    ],
    max_image_bytes: Annotated[int, Depends(get_max_image_bytes)],
) -> ImageClaimExtractionResponse:
    """Extract an assertion from an image; do not verify or persist it."""
    response.headers["Cache-Control"] = "no-store"
    try:
        image_bytes = await image.read(max_image_bytes + 1)
    finally:
        await image.close()
    if image.content_type not in _SUPPORTED_IMAGE_MEDIA_TYPES:
        result = ImageClaimExtractionResult(
            ocr_raw_text="",
            ocr_status="failed",
            cleaned_text="",
            extracted_legal_claim=None,
            ocr_confidence=None,
            ocr_languages=_ocr_languages(),
            low_confidence=True,
            extraction_status="failed",
            detected_language=None,
            language_detection_status="not_run",
            language_detection_reason=None,
            error=SafeExtractionError(
                code="unsupported_image_type",
                stage="image_validation",
            ),
        )
    else:
        result = await run_in_threadpool(extractor, image_bytes)
    return ImageClaimExtractionResponse(
        ocr_raw_text=result.ocr_raw_text,
        ocr_status=result.ocr_status,
        cleaned_text=result.cleaned_text,
        extracted_legal_claim=result.extracted_legal_claim,
        ocr_confidence=result.ocr_confidence,
        ocr_languages=list(result.ocr_languages),
        low_confidence=result.low_confidence,
        extraction_status=result.extraction_status,
        detected_language=result.detected_language,
        language_detection_status=result.language_detection_status,
        language_detection_reason=result.language_detection_reason,
        ocr_method=result.ocr_method,
        claim_extraction_method=result.claim_extraction_method,
        error=(
            {"code": result.error.code, "stage": result.error.stage}
            if result.error is not None
            else None
        ),
    )
