"""Development utility for C3 OCR-only or intentional full extraction runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from app.components.c3_misinformation.image_api import (
    get_image_language_identifier,
    get_ocr_runtime_config,
)
from app.components.c3_misinformation.image_claim_service import (
    extract_claim_from_image,
)
from app.components.c3_misinformation.pipeline.language_detection import (
    detect_language,
)
from app.components.c3_misinformation.pipeline.ocr_extractor import (
    ImageProcessingError,
    OCRProviderError,
    PillowImageProcessor,
    TesseractOCRProvider,
    validate_ocr_languages,
)
from app.components.c3_misinformation.pipeline.text_preprocessing import clean_text


def _languages(value: str) -> tuple[str, ...]:
    languages = tuple(item.strip() for item in value.split(",") if item.strip())
    try:
        return validate_ocr_languages(languages)
    except OCRProviderError as exc:
        raise argparse.ArgumentTypeError(
            "languages must be a comma-separated selection of eng,sin,tam"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="JPEG or PNG image path")
    parser.add_argument(
        "--full",
        action="store_true",
        help="also run Gemini legal-claim extraction (may make a billable request)",
    )
    parser.add_argument("--languages", type=_languages)
    parser.add_argument("--scale", type=int, choices=(1, 2))
    parser.add_argument("--psm", type=int, choices=(6, 11))
    return parser


def _print_ocr_result(
    ocr,
    *,
    cleaned_text: str,
    languages: tuple[str, ...],
    psm: int,
    scale: int,
    language_identifier,
) -> None:
    detection = detect_language(
        cleaned_text,
        language_identifier=language_identifier,
    )
    print("OCR Method:", ocr.method)
    print("OCR Languages:", languages)
    print("OCR PSM:", psm)
    print("Image Scale:", scale)
    print("OCR Confidence:", ocr.confidence)
    print("Detected Language:", detection.label)
    print("Language Detection Status:", detection.status)
    print("\n--- Per-line OCR confidence ---")
    for line in ocr.line_confidences:
        print(f"{line.line_number}: {line.confidence!r} | {line.text}")
    print("\n--- Raw OCR text ---")
    print(ocr.raw_text)
    print("\n--- Cleaned OCR text ---")
    print(cleaned_text)


def main() -> int:
    args = _parser().parse_args()
    runtime = get_ocr_runtime_config()
    languages = args.languages or runtime.languages
    psm = args.psm or runtime.psm
    image_config = replace(
        runtime.image_processing,
        scale_factor=(
            args.scale
            if args.scale is not None
            else runtime.image_processing.scale_factor
        ),
    )
    language_identifier = get_image_language_identifier()

    try:
        with args.image.open("rb") as image_file:
            image_bytes = image_file.read(image_config.max_file_bytes + 1)
    except OSError as exc:
        print(f"Unable to read image: {exc}")
        return 2

    if not args.full:
        try:
            processed = PillowImageProcessor().preprocess(
                image_bytes, config=image_config
            )
            ocr = TesseractOCRProvider(psm=psm).extract(
                processed, languages=languages
            )
        except (ImageProcessingError, OCRProviderError) as exc:
            print("OCR failed safely:", exc.reason)
            return 1
        _print_ocr_result(
            ocr,
            cleaned_text=clean_text(ocr.raw_text, ocr_mode=True),
            languages=languages,
            psm=psm,
            scale=image_config.scale_factor,
            language_identifier=language_identifier,
        )
        if not ocr.raw_text.strip():
            print("OCR failed safely: empty_ocr_result")
            return 1
        return 0

    from app.components.c3_misinformation.pipeline.claim_extraction import (
        GeminiClaimExtractionProvider,
    )

    result = extract_claim_from_image(
        image_bytes,
        ocr_provider=TesseractOCRProvider(psm=psm),
        ocr_languages=languages,
        claim_provider=GeminiClaimExtractionProvider(),
        image_config=image_config,
        low_confidence_threshold=runtime.low_confidence_threshold,
        language_identifier=language_identifier,
    )
    print("OCR Status:", result.ocr_status)
    print("OCR Confidence:", result.ocr_confidence)
    print("OCR Languages:", result.ocr_languages)
    print("Detected Language:", result.detected_language)
    print("Language Detection Status:", result.language_detection_status)
    print("Extraction Status:", result.extraction_status)
    print("Extracted Claim:", result.extracted_legal_claim)
    print("Error:", result.error)
    print("\n--- Per-line OCR confidence ---")
    for line in result.ocr_line_confidences:
        print(f"{line.line_number}: {line.confidence!r} | {line.text}")
    print("\n--- Raw OCR text ---")
    print(result.ocr_raw_text)
    print("\n--- Cleaned OCR text ---")
    print(result.cleaned_text)
    return 0 if result.ocr_status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
