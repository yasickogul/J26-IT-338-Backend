"""C3-owned multipart API for OCR claim extraction only."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Response, UploadFile

from .config import llm_provider
from .image_api_schemas import ImageClaimExtractionResponse
from .image_claim_service import (
    ImageClaimExtractionResult,
    SafeExtractionError,
    extract_claim_from_image,
)
from .pipeline.claim_extraction import GeminiClaimExtractionProvider
from .pipeline.fasttext_language_identifier import (
    FastTextLanguageIdentifier,
    get_fasttext_language_identifier,
)
from .pipeline.ocr_extractor import (
    SUPPORTED_OCR_LANGUAGES,
    ImageProcessingConfig,
    TesseractOCRProvider,
)
from .pipeline.qwen_claim_extraction import QwenClaimExtractionProvider

router = APIRouter(tags=["C3 image claim extraction"])
ImageClaimExtractor = Callable[[bytes], ImageClaimExtractionResult]
_SUPPORTED_IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png"}

_DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024
_DEFAULT_MAX_DECODED_PIXELS = 40_000_000
_DEFAULT_MAX_PROCESSED_PIXELS = 40_000_000


async def _run_c3_blocking(callable_: Callable[..., object], *args: object) -> object:
    """Run one blocking image boundary without the unstable default executor."""
    asyncio.get_running_loop()
    value: object = None
    error: BaseException | None = None
    finished = threading.Event()

    def complete() -> None:
        nonlocal value, error
        try:
            value = callable_(*args)
        except BaseException as exc:  # noqa: BLE001 - transfer thread outcome
            error = exc
        finally:
            finished.set()

    thread = threading.Thread(target=complete, name="c3-image-boundary", daemon=True)
    thread.start()
    try:
        while not finished.is_set():
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        raise
    thread.join(timeout=0.1)
    if error is not None:
        raise error
    return value


@dataclass(frozen=True)
class OCRRuntimeConfig:
    """Validated deployment settings shared by the API and development tools."""

    image_processing: ImageProcessingConfig
    languages: tuple[str, ...]
    low_confidence_threshold: float
    psm: Literal[6, 11]


def _positive_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _boolean_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = raw_value.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _bounded_env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def get_max_image_bytes() -> int:
    return _positive_env_int("C3_OCR_MAX_IMAGE_BYTES", _DEFAULT_MAX_FILE_BYTES)


async def get_max_image_bytes_dependency() -> int:
    """Async FastAPI dependency that avoids the default executor bridge."""
    return get_max_image_bytes()


def _confidence_threshold() -> float:
    value = float(os.getenv("C3_OCR_LOW_CONFIDENCE_THRESHOLD", "0.60"))
    if not 0 <= value <= 1:
        raise ValueError("C3_OCR_LOW_CONFIDENCE_THRESHOLD must be between 0 and 1")
    return value


def _ocr_languages() -> tuple[str, ...]:
    configured = tuple(
        item.strip()
        for item in os.getenv("C3_OCR_LANGUAGES", "eng,sin,tam").split(",")
        if item.strip()
    )
    if not configured:
        raise ValueError("C3_OCR_LANGUAGES must contain at least one language")
    unsupported = tuple(
        language
        for language in configured
        if language not in SUPPORTED_OCR_LANGUAGES
    )
    if unsupported:
        raise ValueError(
            "C3_OCR_LANGUAGES supports only eng, sin, and tam"
        )
    return tuple(dict.fromkeys(configured))


def _ocr_image_scale() -> int:
    scale = _positive_env_int("C3_OCR_IMAGE_SCALE", 1)
    if scale not in (1, 2):
        raise ValueError("C3_OCR_IMAGE_SCALE must be 1 or 2")
    return scale


def _ocr_psm() -> int:
    psm = _positive_env_int("C3_OCR_PSM", 6)
    if psm not in (6, 11):
        raise ValueError("C3_OCR_PSM must be 6 or 11")
    return psm


def get_ocr_runtime_config() -> OCRRuntimeConfig:
    """Load and validate C3 OCR environment configuration."""
    image_processing = ImageProcessingConfig(
        max_file_bytes=get_max_image_bytes(),
        max_decoded_pixels=_positive_env_int(
            "C3_OCR_MAX_DECODED_PIXELS",
            _DEFAULT_MAX_DECODED_PIXELS,
        ),
        max_processed_pixels=_positive_env_int(
            "C3_OCR_MAX_PROCESSED_PIXELS",
            _DEFAULT_MAX_PROCESSED_PIXELS,
        ),
        scale_factor=_ocr_image_scale(),
        improve_contrast=_boolean_env("C3_OCR_IMPROVE_CONTRAST", True),
        reduce_noise=_boolean_env("C3_OCR_REDUCE_NOISE", False),
        deskew=_boolean_env("C3_OCR_DESKEW", False),
        deskew_max_angle_degrees=_bounded_env_float(
            "C3_OCR_DESKEW_MAX_ANGLE_DEGREES",
            5.0,
            minimum=1.0,
            maximum=10.0,
        ),
    )
    return OCRRuntimeConfig(
        image_processing=image_processing,
        languages=_ocr_languages(),
        low_confidence_threshold=_confidence_threshold(),
        psm=_ocr_psm(),
    )


@lru_cache(maxsize=1)
def get_image_language_identifier() -> FastTextLanguageIdentifier:
    """Return one lazy local-only identifier shared by image requests."""
    return get_fasttext_language_identifier()


@lru_cache(maxsize=1)
def get_image_claim_extractor() -> ImageClaimExtractor:
    runtime = get_ocr_runtime_config()
    ocr_provider = TesseractOCRProvider(psm=runtime.psm)
    claim_provider = (
        QwenClaimExtractionProvider() if llm_provider() == "qwen"
        else GeminiClaimExtractionProvider()
    )
    language_identifier = get_image_language_identifier()

    def extractor(image_bytes: bytes) -> ImageClaimExtractionResult:
        return extract_claim_from_image(
            image_bytes,
            ocr_provider=ocr_provider,
            claim_provider=claim_provider,
            image_config=runtime.image_processing,
            ocr_languages=runtime.languages,
            low_confidence_threshold=runtime.low_confidence_threshold,
            language_identifier=language_identifier,
        )

    return extractor


async def get_image_claim_extractor_dependency() -> ImageClaimExtractor:
    """Async FastAPI dependency for the cached C3 extractor factory."""
    return get_image_claim_extractor()


@router.post(
    "/claims/extract-image",
    response_model=ImageClaimExtractionResponse,
)
async def extract_image_claim(
    image: Annotated[UploadFile, File()],
    response: Response,
    extractor: Annotated[
        ImageClaimExtractor,
        Depends(get_image_claim_extractor_dependency),
    ],
    max_image_bytes: Annotated[int, Depends(get_max_image_bytes_dependency)],
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
        result = await _run_c3_blocking(extractor, image_bytes)
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
        ocr_line_confidences=[
            {
                "line_number": line.line_number,
                "text": line.text,
                "confidence": line.confidence,
            }
            for line in result.ocr_line_confidences
        ],
        error=(
            {"code": result.error.code, "stage": result.error.stage}
            if result.error is not None
            else None
        ),
    )
