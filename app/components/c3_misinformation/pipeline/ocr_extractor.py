"""Replaceable image preprocessing and OCR contracts for legal claim images."""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
from typing import Any, Literal, Protocol

ImageFailureReason = Literal[
    "empty_image",
    "image_too_large",
    "unsupported_image_type",
    "corrupted_image",
    "decoded_image_too_large",
    "image_processing_failed",
]
OCRFailureReason = Literal[
    "ocr_provider_unavailable",
    "ocr_language_unavailable",
    "ocr_provider_error",
    "invalid_ocr_response",
]
SUPPORTED_OCR_LANGUAGES = frozenset({"eng", "sin", "tam"})
_DESKEW_SAMPLE_MAX_DIMENSION = 512
_DESKEW_ANGLE_STEP_DEGREES = 0.5
_DESKEW_MIN_SCORE_IMPROVEMENT = 0.08
_DESKEW_MIN_FOREGROUND_PIXELS = 64


class ImageProcessingError(RuntimeError):
    """An image validation/preprocessing failure safe to expose by code."""

    def __init__(self, reason: ImageFailureReason) -> None:
        super().__init__(reason)
        self.reason = reason


class OCRProviderError(RuntimeError):
    """An OCR provider failure safe to expose by code."""

    def __init__(self, reason: OCRFailureReason) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ImageProcessingConfig:
    """Operational upload limits; values are deployment config, not SRDS limits."""

    max_file_bytes: int
    max_decoded_pixels: int
    improve_contrast: bool = True
    reduce_noise: bool = False
    max_processed_pixels: int = 40_000_000
    scale_factor: Literal[1, 2] = 1
    deskew: bool = False
    deskew_max_angle_degrees: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.max_file_bytes, bool) or self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        if isinstance(self.max_decoded_pixels, bool) or self.max_decoded_pixels <= 0:
            raise ValueError("max_decoded_pixels must be a positive integer")
        if (
            isinstance(self.max_processed_pixels, bool)
            or self.max_processed_pixels <= 0
        ):
            raise ValueError("max_processed_pixels must be a positive integer")
        if (
            isinstance(self.scale_factor, bool)
            or not isinstance(self.scale_factor, int)
            or self.scale_factor not in (1, 2)
        ):
            raise ValueError("scale_factor must be 1 or 2")
        if not isinstance(self.deskew, bool):
            raise TypeError("deskew must be a boolean")
        if (
            isinstance(self.deskew_max_angle_degrees, bool)
            or not isinstance(self.deskew_max_angle_degrees, (int, float))
            or not math.isfinite(self.deskew_max_angle_degrees)
            or not 1 <= self.deskew_max_angle_degrees <= 10
        ):
            raise ValueError(
                "deskew_max_angle_degrees must be between 1 and 10"
            )


@dataclass(frozen=True)
class ProcessedImage:
    data: bytes
    source_format: Literal["JPEG", "PNG"]
    width: int
    height: int


@dataclass(frozen=True)
class OCRLineConfidence:
    """Confidence for one OCR output line; this is not legal confidence."""

    line_number: int
    text: str
    confidence: float | None


@dataclass(frozen=True)
class OCRProviderResult:
    raw_text: str
    confidence: float | None
    method: str
    line_confidences: tuple[OCRLineConfidence, ...] = ()


class ImageProcessor(Protocol):
    def preprocess(
        self,
        image_bytes: bytes,
        *,
        config: ImageProcessingConfig,
    ) -> ProcessedImage:
        """Validate and normalize an uploaded JPEG or PNG."""


class OCRProvider(Protocol):
    def extract(
        self,
        image: ProcessedImage,
        *,
        languages: tuple[str, ...],
    ) -> OCRProviderResult:
        """Extract text in reading order and report confidence when available."""


def validate_image_content(
    image_bytes: bytes,
    *,
    config: ImageProcessingConfig,
) -> tuple[Literal["JPEG", "PNG"], int, int]:
    """Inspect decoded image metadata rather than trusting names or media types."""
    if not isinstance(image_bytes, bytes):
        raise TypeError("image_bytes must be bytes")
    if not image_bytes:
        raise ImageProcessingError("empty_image")
    if len(image_bytes) > config.max_file_bytes:
        raise ImageProcessingError("image_too_large")
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        source_format: Literal["JPEG", "PNG"] = "PNG"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        source_format = "JPEG"
    else:
        raise ImageProcessingError("unsupported_image_type")
    try:
        import pymupdf
    except ImportError:
        raise ImageProcessingError("image_processing_failed") from None
    try:
        with pymupdf.open(stream=image_bytes) as document:
            if document.page_count != 1:
                raise ImageProcessingError("corrupted_image")
            page = document[0]
            image_info = page.get_image_info()
            if len(image_info) != 1:
                raise ImageProcessingError("corrupted_image")
            width = image_info[0].get("width")
            height = image_info[0].get("height")
            if (
                isinstance(width, bool)
                or not isinstance(width, int)
                or isinstance(height, bool)
                or not isinstance(height, int)
                or width <= 0
                or height <= 0
            ):
                raise ImageProcessingError("corrupted_image")
            if width * height > config.max_decoded_pixels:
                raise ImageProcessingError("decoded_image_too_large")
    except ImageProcessingError:
        raise
    except Exception:  # noqa: BLE001 - third-party decoder boundary
        raise ImageProcessingError("corrupted_image") from None
    return source_format, width, height


def _load_pillow() -> tuple[Any, Any, Any, Any]:
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError:
        raise ImageProcessingError("image_processing_failed") from None
    return Image, ImageEnhance, ImageFilter, ImageOps


def _otsu_threshold(image: Any) -> int:
    histogram = image.histogram()
    total = sum(histogram)
    weighted_total = sum(index * count for index, count in enumerate(histogram))
    background_count = 0
    background_weight = 0
    best_variance = -1.0
    best_threshold = 127
    for threshold, count in enumerate(histogram):
        background_count += count
        if background_count == 0:
            continue
        foreground_count = total - background_count
        if foreground_count == 0:
            break
        background_weight += threshold * count
        background_mean = background_weight / background_count
        foreground_mean = (
            weighted_total - background_weight
        ) / foreground_count
        variance = (
            background_count
            * foreground_count
            * (background_mean - foreground_mean) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def _horizontal_projection_score(binary_image: Any) -> float:
    width, height = binary_image.size
    row_counts = [
        binary_image.crop((0, row, width, row + 1)).histogram()[255]
        for row in range(height)
    ]
    return float(
        sum(
            (current - previous) ** 2
            for previous, current in pairwise(row_counts)
        )
    )


def _estimate_deskew_angle(
    grayscale: Any,
    *,
    Image: Any,
    ImageOps: Any,
    max_angle_degrees: float,
) -> float | None:
    """Estimate a small correction angle from horizontal text projections."""
    sample = ImageOps.autocontrast(grayscale)
    if max(sample.size) > _DESKEW_SAMPLE_MAX_DIMENSION:
        ratio = _DESKEW_SAMPLE_MAX_DIMENSION / max(sample.size)
        sample = sample.resize(
            (
                max(1, round(sample.width * ratio)),
                max(1, round(sample.height * ratio)),
            ),
            resample=Image.Resampling.BILINEAR,
        )
    threshold = _otsu_threshold(sample)
    binary = sample.point(
        [255 if value <= threshold else 0 for value in range(256)],
        mode="L",
    )
    foreground_pixels = binary.histogram()[255]
    total_pixels = binary.width * binary.height
    if (
        foreground_pixels < _DESKEW_MIN_FOREGROUND_PIXELS
        or foreground_pixels > total_pixels * 0.5
    ):
        return None

    baseline_score = _horizontal_projection_score(binary)
    if baseline_score <= 0:
        return None
    steps = int(max_angle_degrees / _DESKEW_ANGLE_STEP_DEGREES)
    candidates = (
        step * _DESKEW_ANGLE_STEP_DEGREES
        for step in range(-steps, steps + 1)
    )
    best_angle = 0.0
    best_score = baseline_score
    for angle in candidates:
        if angle == 0:
            continue
        rotated = binary.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            expand=False,
            fillcolor=0,
        )
        score = _horizontal_projection_score(rotated)
        if score > best_score:
            best_angle = angle
            best_score = score

    improvement = (best_score - baseline_score) / baseline_score
    if (
        best_angle == 0
        or abs(best_angle) >= max_angle_degrees
        or improvement < _DESKEW_MIN_SCORE_IMPROVEMENT
    ):
        return None
    return best_angle


class PillowImageProcessor:
    """Decode, orient, and normalize images without retaining the upload."""

    def preprocess(
        self,
        image_bytes: bytes,
        *,
        config: ImageProcessingConfig,
    ) -> ProcessedImage:
        validated_format, validated_width, validated_height = validate_image_content(
            image_bytes, config=config
        )

        Image, ImageEnhance, ImageFilter, ImageOps = _load_pillow()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(image_bytes)) as probe:
                    source_format = (probe.format or "").upper()
                    if source_format != validated_format:
                        raise ImageProcessingError("corrupted_image")
                    width, height = probe.size
                    if (width, height) != (validated_width, validated_height):
                        raise ImageProcessingError("corrupted_image")
                    probe.verify()

                with Image.open(BytesIO(image_bytes)) as decoded:
                    decoded.load()
                    oriented = ImageOps.exif_transpose(decoded)
                    if "A" in oriented.getbands():
                        background = Image.new("RGBA", oriented.size, "white")
                        background.alpha_composite(oriented.convert("RGBA"))
                        oriented = background.convert("RGB")
                    grayscale = ImageOps.grayscale(oriented)
                    if config.deskew:
                        correction_angle = _estimate_deskew_angle(
                            grayscale,
                            Image=Image,
                            ImageOps=ImageOps,
                            max_angle_degrees=config.deskew_max_angle_degrees,
                        )
                        if correction_angle is not None:
                            grayscale = grayscale.rotate(
                                correction_angle,
                                resample=Image.Resampling.BICUBIC,
                                expand=True,
                                fillcolor=255,
                            )
                            if (
                                grayscale.width * grayscale.height
                                > config.max_processed_pixels
                            ):
                                raise ImageProcessingError(
                                    "decoded_image_too_large"
                                )
                    scaled_width = grayscale.width * config.scale_factor
                    scaled_height = grayscale.height * config.scale_factor
                    if scaled_width * scaled_height > config.max_processed_pixels:
                        raise ImageProcessingError("decoded_image_too_large")
                    if config.scale_factor != 1:
                        grayscale = grayscale.resize(
                            (scaled_width, scaled_height),
                            resample=Image.Resampling.LANCZOS,
                        )
                    if config.improve_contrast:
                        grayscale = ImageEnhance.Contrast(grayscale).enhance(1.25)
                    if config.reduce_noise:
                        grayscale = grayscale.filter(ImageFilter.MedianFilter(size=3))
                    output = BytesIO()
                    grayscale.save(output, format="PNG", optimize=False)
                    final_width, final_height = grayscale.size
        except ImageProcessingError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ImageProcessingError("decoded_image_too_large") from None
        except (OSError, SyntaxError, ValueError):
            raise ImageProcessingError("corrupted_image") from None
        except Exception:  # noqa: BLE001 - third-party decoder boundary
            raise ImageProcessingError("image_processing_failed") from None

        return ProcessedImage(
            data=output.getvalue(),
            source_format=source_format,
            width=final_width,
            height=final_height,
        )


def validate_ocr_languages(languages: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and deduplicate the C3-supported Tesseract language packs."""
    if not isinstance(languages, tuple) or not languages:
        raise OCRProviderError("invalid_ocr_response")
    normalized: list[str] = []
    for language in languages:
        if not isinstance(language, str) or not re.fullmatch(
            r"[A-Za-z0-9_]+", language
        ):
            raise OCRProviderError("invalid_ocr_response")
        if language not in SUPPORTED_OCR_LANGUAGES:
            raise OCRProviderError("ocr_language_unavailable")
        if language not in normalized:
            normalized.append(language)
    return tuple(normalized)


class TesseractOCRProvider:
    """Optional pytesseract adapter; no executable or model is downloaded."""

    def __init__(self, *, psm: int = 6) -> None:
        if isinstance(psm, bool) or not isinstance(psm, int) or psm not in (6, 11):
            raise ValueError("psm must be 6 or 11")
        self.psm = psm

    def extract(
        self,
        image: ProcessedImage,
        *,
        languages: tuple[str, ...],
    ) -> OCRProviderResult:
        configured_languages = validate_ocr_languages(languages)
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise OCRProviderError("ocr_provider_unavailable") from None

        try:
            installed = set(pytesseract.get_languages(config=""))
        except getattr(pytesseract, "TesseractNotFoundError", RuntimeError):
            raise OCRProviderError("ocr_provider_unavailable") from None
        except Exception:  # noqa: BLE001 - third-party executable boundary
            raise OCRProviderError("ocr_provider_error") from None
        if not set(configured_languages).issubset(installed):
            raise OCRProviderError("ocr_language_unavailable")

        try:
            with Image.open(BytesIO(image.data)) as prepared:
                data = pytesseract.image_to_data(
                    prepared,
                    lang="+".join(configured_languages),
                    config=f"--psm {self.psm}",
                    output_type=pytesseract.Output.DICT,
                )
        except getattr(pytesseract, "TesseractNotFoundError", RuntimeError):
            raise OCRProviderError("ocr_provider_unavailable") from None
        except getattr(pytesseract, "TesseractError", RuntimeError) as exc:
            message = str(exc).casefold()
            reason: OCRFailureReason = (
                "ocr_language_unavailable"
                if "traineddata" in message or "language" in message
                else "ocr_provider_error"
            )
            raise OCRProviderError(reason) from None
        except Exception:  # noqa: BLE001 - third-party OCR boundary
            raise OCRProviderError("ocr_provider_error") from None

        if not isinstance(data, dict):
            raise OCRProviderError("invalid_ocr_response")
        texts = data.get("text")
        confidences = data.get("conf")
        if not isinstance(texts, list) or not isinstance(confidences, list):
            raise OCRProviderError("invalid_ocr_response")

        words: list[str] = []
        scores: list[float] = []
        line_words: list[str] = []
        line_scores: list[float] = []
        line_confidences: list[OCRLineConfidence] = []
        previous_line: tuple[Any, Any, Any] | None = None
        blocks = data.get("block_num", [])
        paragraphs = data.get("par_num", [])
        lines = data.get("line_num", [])
        for index, value in enumerate(texts):
            word = value.strip() if isinstance(value, str) else ""
            if not word:
                continue
            line_key = (
                blocks[index] if index < len(blocks) else None,
                paragraphs[index] if index < len(paragraphs) else None,
                lines[index] if index < len(lines) else None,
            )
            if words and line_key != previous_line:
                line_confidences.append(
                    OCRLineConfidence(
                        line_number=len(line_confidences) + 1,
                        text=" ".join(line_words),
                        confidence=(
                            sum(line_scores) / len(line_scores)
                            if line_scores
                            else None
                        ),
                    )
                )
                line_words = []
                line_scores = []
                words.append("\n")
            elif words:
                words.append(" ")
            words.append(word)
            line_words.append(word)
            previous_line = line_key
            try:
                score = float(confidences[index])
            except (IndexError, TypeError, ValueError):
                continue
            if math.isfinite(score) and 0 <= score <= 100:
                normalized_score = score / 100.0
                scores.append(normalized_score)
                line_scores.append(normalized_score)

        if line_words:
            line_confidences.append(
                OCRLineConfidence(
                    line_number=len(line_confidences) + 1,
                    text=" ".join(line_words),
                    confidence=(
                        sum(line_scores) / len(line_scores)
                        if line_scores
                        else None
                    ),
                )
            )

        return OCRProviderResult(
            raw_text="".join(words).strip(),
            confidence=(sum(scores) / len(scores)) if scores else None,
            method="tesseract",
            line_confidences=tuple(line_confidences),
        )
