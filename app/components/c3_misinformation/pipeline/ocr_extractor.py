"""Replaceable image preprocessing and OCR contracts for legal claim images."""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from io import BytesIO
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

    def __post_init__(self) -> None:
        if isinstance(self.max_file_bytes, bool) or self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        if isinstance(self.max_decoded_pixels, bool) or self.max_decoded_pixels <= 0:
            raise ValueError("max_decoded_pixels must be a positive integer")


@dataclass(frozen=True)
class ProcessedImage:
    data: bytes
    source_format: Literal["JPEG", "PNG"]
    width: int
    height: int


@dataclass(frozen=True)
class OCRProviderResult:
    raw_text: str
    confidence: float | None
    method: str


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


def _validated_languages(languages: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(languages, tuple) or not languages:
        raise OCRProviderError("invalid_ocr_response")
    normalized: list[str] = []
    for language in languages:
        if not isinstance(language, str) or not re.fullmatch(
            r"[A-Za-z0-9_]+", language
        ):
            raise OCRProviderError("invalid_ocr_response")
        if language not in normalized:
            normalized.append(language)
    return tuple(normalized)


class TesseractOCRProvider:
    """Optional pytesseract adapter; no executable or model is downloaded."""

    def extract(
        self,
        image: ProcessedImage,
        *,
        languages: tuple[str, ...],
    ) -> OCRProviderResult:
        configured_languages = _validated_languages(languages)
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
                    config="--psm 6",
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
                words.append("\n")
            elif words:
                words.append(" ")
            words.append(word)
            previous_line = line_key
            try:
                score = float(confidences[index])
            except (IndexError, TypeError, ValueError):
                continue
            if math.isfinite(score) and 0 <= score <= 100:
                scores.append(score / 100.0)

        return OCRProviderResult(
            raw_text="".join(words).strip(),
            confidence=(sum(scores) / len(scores)) if scores else None,
            method="tesseract",
        )
