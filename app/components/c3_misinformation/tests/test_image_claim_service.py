"""Deterministic unit tests for the image-to-claim extraction pipeline."""

from dataclasses import dataclass
from io import BytesIO
from unittest import TestCase
from unittest.mock import patch

import pymupdf

from app.components.c3_misinformation.image_claim_service import (
    extract_claim_from_image,
)
from app.components.c3_misinformation.pipeline.claim_extraction import (
    ClaimExtractionProviderError,
    ClaimExtractionProviderResponse,
)
from app.components.c3_misinformation.pipeline.language_detection import (
    LanguageIdentification,
)
from app.components.c3_misinformation.pipeline.ocr_extractor import (
    ImageProcessingConfig,
    ImageProcessingError,
    OCRLineConfidence,
    OCRProviderError,
    OCRProviderResult,
    PillowImageProcessor,
    ProcessedImage,
    TesseractOCRProvider,
    validate_image_content,
)

CONFIG = ImageProcessingConfig(max_file_bytes=100, max_decoded_pixels=1_000)


class DeterministicImageProcessor:
    """A decoder double used so tests need no Pillow or OCR installation."""

    def preprocess(self, image_bytes, *, config):
        if not image_bytes:
            raise ImageProcessingError("empty_image")
        if len(image_bytes) > config.max_file_bytes:
            raise ImageProcessingError("image_too_large")
        if image_bytes.startswith(b"huge-pixels"):
            raise ImageProcessingError("decoded_image_too_large")
        if image_bytes.startswith(b"\xff\xd8\xff"):
            image_format = "JPEG"
        elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            image_format = "PNG"
        elif image_bytes.startswith(b"corrupt"):
            raise ImageProcessingError("corrupted_image")
        else:
            raise ImageProcessingError("unsupported_image_type")
        return ProcessedImage(
            data=b"normalized",
            source_format=image_format,
            width=20,
            height=10,
        )


@dataclass
class StubOCRProvider:
    result: OCRProviderResult | None = None
    error: str | None = None

    def extract(self, image, *, languages):
        if self.error:
            raise OCRProviderError(self.error)
        return self.result


@dataclass
class StubClaimProvider:
    status: str = "extracted"
    extracted_claim: str | None = "A legal claim."
    error: str | None = None

    def extract(self, cleaned_text):
        if self.error:
            raise ClaimExtractionProviderError(self.error)
        return ClaimExtractionProviderResponse(
            status=self.status,
            extracted_claim=self.extracted_claim,
            method="stub-claim-provider",
        )


def run_pipeline(
    *,
    image=b"\x89PNG\r\n\x1a\nvalid",
    text="A legal claim.",
    confidence=0.95,
    claim_provider=None,
    ocr_error=None,
    line_confidences=(),
    language_identifier=None,
):
    return extract_claim_from_image(
        image,
        ocr_provider=StubOCRProvider(
            result=OCRProviderResult(
                text,
                confidence,
                "stub-ocr",
                line_confidences=line_confidences,
            ),
            error=ocr_error,
        ),
        claim_provider=claim_provider or StubClaimProvider(),
        image_config=CONFIG,
        image_processor=DeterministicImageProcessor(),
        language_identifier=language_identifier,
    )


class ImageClaimServiceTests(TestCase):
    def test_real_preprocessor_corrects_orientation_and_outputs_grayscale(self):
        from PIL import Image

        image = Image.new("RGB", (4, 2), "white")
        exif = Image.Exif()
        exif[274] = 6
        source = BytesIO()
        image.save(source, format="JPEG", exif=exif)

        result = PillowImageProcessor().preprocess(
            source.getvalue(),
            config=ImageProcessingConfig(
                max_file_bytes=10_000,
                max_decoded_pixels=1_000,
                improve_contrast=True,
                reduce_noise=True,
            ),
        )

        self.assertEqual(result.source_format, "JPEG")
        self.assertEqual((result.width, result.height), (2, 4))
        with Image.open(BytesIO(result.data)) as prepared:
            self.assertEqual(prepared.format, "PNG")
            self.assertEqual(prepared.mode, "L")

    def test_real_preprocessor_supports_two_x_enlargement(self):
        from PIL import Image

        source = BytesIO()
        Image.new("RGB", (4, 3), "white").save(source, format="PNG")
        result = PillowImageProcessor().preprocess(
            source.getvalue(),
            config=ImageProcessingConfig(
                max_file_bytes=10_000,
                max_decoded_pixels=1_000,
                max_processed_pixels=100,
                scale_factor=2,
                improve_contrast=False,
            ),
        )
        self.assertEqual((result.width, result.height), (8, 6))

    def test_optional_deskew_expands_canvas_without_cropping(self):
        from PIL import Image

        source = BytesIO()
        Image.new("L", (20, 10), "white").save(source, format="PNG")
        with patch(
            "app.components.c3_misinformation.pipeline.ocr_extractor."
            "_estimate_deskew_angle",
            return_value=3.0,
        ):
            result = PillowImageProcessor().preprocess(
                source.getvalue(),
                config=ImageProcessingConfig(
                    max_file_bytes=10_000,
                    max_decoded_pixels=1_000,
                    max_processed_pixels=1_000,
                    improve_contrast=False,
                    deskew=True,
                ),
            )
        self.assertGreater(result.width, 20)
        self.assertGreater(result.height, 10)

    def test_deskew_rotation_obeys_processed_pixel_limit(self):
        from PIL import Image

        source = BytesIO()
        Image.new("L", (20, 10), "white").save(source, format="PNG")
        with (
            patch(
                "app.components.c3_misinformation.pipeline.ocr_extractor."
                "_estimate_deskew_angle",
                return_value=3.0,
            ),
            self.assertRaises(ImageProcessingError) as raised,
        ):
            PillowImageProcessor().preprocess(
                source.getvalue(),
                config=ImageProcessingConfig(
                    max_file_bytes=10_000,
                    max_decoded_pixels=1_000,
                    max_processed_pixels=200,
                    deskew=True,
                ),
            )
        self.assertEqual(raised.exception.reason, "decoded_image_too_large")

    def test_real_preprocessor_rejects_enlargement_over_processed_pixel_limit(self):
        from PIL import Image

        source = BytesIO()
        Image.new("RGB", (4, 3), "white").save(source, format="PNG")
        with self.assertRaises(ImageProcessingError) as raised:
            PillowImageProcessor().preprocess(
                source.getvalue(),
                config=ImageProcessingConfig(
                    max_file_bytes=10_000,
                    max_decoded_pixels=1_000,
                    max_processed_pixels=47,
                    scale_factor=2,
                ),
            )
        self.assertEqual(raised.exception.reason, "decoded_image_too_large")

    def test_real_preprocessor_composites_transparent_png_on_white(self):
        from PIL import Image

        image = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
        source = BytesIO()
        image.save(source, format="PNG")
        result = PillowImageProcessor().preprocess(
            source.getvalue(),
            config=ImageProcessingConfig(
                max_file_bytes=10_000,
                max_decoded_pixels=1_000,
                improve_contrast=False,
            ),
        )
        with Image.open(BytesIO(result.data)) as prepared:
            self.assertEqual(
                {
                    prepared.getpixel((x, y))
                    for y in range(prepared.height)
                    for x in range(prepared.width)
                },
                {255},
            )

    def test_real_jpeg_and_png_content_is_validated(self):
        validation_config = ImageProcessingConfig(
            max_file_bytes=10_000,
            max_decoded_pixels=1_000,
        )
        pixmap = pymupdf.Pixmap(
            pymupdf.csRGB,
            pymupdf.IRect(0, 0, 2, 3),
            False,
        )
        pixmap.clear_with(255)
        for output_format, expected in (("jpeg", "JPEG"), ("png", "PNG")):
            with self.subTest(output_format=output_format):
                image_bytes = pixmap.tobytes(output_format)
                self.assertEqual(
                    validate_image_content(image_bytes, config=validation_config),
                    (expected, 2, 3),
                )

    def test_real_validator_rejects_disguised_and_corrupted_images(self):
        for image in (b"GIF89a", b"\x89PNG\r\n\x1a\ncorrupt"):
            with self.subTest(image=image[:8]):
                with self.assertRaises(ImageProcessingError) as raised:
                    validate_image_content(image, config=CONFIG)
                self.assertIn(
                    raised.exception.reason,
                    {"unsupported_image_type", "corrupted_image"},
                )

    def test_real_validator_rejects_excessive_decoded_dimensions(self):
        from PIL import Image

        source = BytesIO()
        Image.new("RGB", (11, 10), "white").save(source, format="PNG")
        with self.assertRaises(ImageProcessingError) as raised:
            validate_image_content(
                source.getvalue(),
                config=ImageProcessingConfig(
                    max_file_bytes=10_000,
                    max_decoded_pixels=100,
                ),
            )
        self.assertEqual(raised.exception.reason, "decoded_image_too_large")

    def test_valid_jpeg_and_png_are_processed_by_actual_content(self):
        for image in (b"\xff\xd8\xffvalid-jpeg", b"\x89PNG\r\n\x1a\nvalid-png"):
            with self.subTest(image=image[:8]):
                result = run_pipeline(image=image)
                self.assertEqual(result.extraction_status, "extracted")
                self.assertEqual(result.ocr_raw_text, "A legal claim.")

    def test_unsupported_corrupted_oversized_and_large_decoded_images(self):
        cases = (
            (b"", "empty_image"),
            (b"GIF89a", "unsupported_image_type"),
            (b"corrupt-image", "corrupted_image"),
            (b"x" * 101, "image_too_large"),
            (b"huge-pixels", "decoded_image_too_large"),
        )
        for image, code in cases:
            with self.subTest(code=code):
                result = run_pipeline(image=image)
                self.assertEqual(result.extraction_status, "failed")
                self.assertEqual(result.error.code, code)
                self.assertEqual(result.error.stage, "image_validation")

    def test_empty_ocr_output_is_explicit_and_does_not_call_claim_provider(self):
        class UnexpectedClaimProvider:
            def extract(self, cleaned_text):
                raise AssertionError("claim extraction must not run")

        result = run_pipeline(text=" \n ", claim_provider=UnexpectedClaimProvider())
        self.assertEqual(result.error.code, "empty_ocr_result")
        self.assertEqual(result.error.stage, "ocr")
        self.assertEqual(result.extraction_status, "failed")

    def test_low_confidence_and_unknown_confidence_are_flagged(self):
        self.assertTrue(run_pipeline(confidence=0.59).low_confidence)
        self.assertFalse(run_pipeline(confidence=0.60).low_confidence)
        self.assertTrue(run_pipeline(confidence=None).low_confidence)

    def test_per_line_confidence_is_preserved_separately(self):
        lines = (
            OCRLineConfidence(1, "Section 10", 0.9),
            OCRLineConfidence(2, "applies", None),
        )
        result = run_pipeline(
            text="Section 10\napplies",
            line_confidences=lines,
        )
        self.assertEqual(result.ocr_line_confidences, lines)

    def test_ocr_provider_unavailable_is_sanitized(self):
        result = run_pipeline(ocr_error="ocr_provider_unavailable")
        self.assertEqual(result.error.code, "ocr_provider_unavailable")
        self.assertEqual(result.error.stage, "ocr")
        self.assertEqual(result.ocr_raw_text, "")

    def test_ocr_provider_error_is_sanitized(self):
        result = run_pipeline(ocr_error="ocr_provider_error")
        self.assertEqual(result.error.code, "ocr_provider_error")
        self.assertEqual(result.error.stage, "ocr")

    def test_english_sinhala_tamil_and_mixed_text_are_preserved(self):
        texts = (
            "A tenant must give notice under section 10.",
            "\u0db1\u0dd3\u0dad\u0dd2\u0dba \u0d85\u0db1\u0dd4\u0dc0 \u0daf\u0dd2\u0db1 30 \u0daf\u0dd0\u0db1\u0dd4\u0db8\u0dca \u0daf\u0dd2\u0dba \u0dba\u0dd4\u0dad\u0dd4\u0dba.",
            "\u0bb5\u0bbe\u0b9f\u0b95\u0bc8\u0baf\u0bbe\u0bb3\u0bb0\u0bcd 30 \u0ba8\u0bbe\u0b9f\u0bcd\u0b95\u0bb3\u0bcd \u0b85\u0bb1\u0bbf\u0bb5\u0bbf\u0baa\u0bcd\u0baa\u0bc1 \u0bb5\u0bb4\u0b99\u0bcd\u0b95 \u0bb5\u0bc7\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd.",
            "Section 10 \u0dba\u0da7\u0dad\u0dda tenant \u0daf\u0dd2\u0db1 30 notice \u0d91\u0d9a\u0d9a\u0dca \u0daf\u0dd2\u0dba \u0dba\u0dd4\u0dad\u0dd4\u0dba.",
        )
        for text in texts:
            with self.subTest(text=text):
                result = run_pipeline(
                    text=f"  {text}\r\n",
                    claim_provider=StubClaimProvider(extracted_claim=text),
                )
                self.assertEqual(result.ocr_raw_text, f"  {text}\r\n")
                self.assertEqual(result.cleaned_text, text)
                self.assertEqual(result.extracted_legal_claim, text)
                self.assertEqual(result.ocr_languages, ("eng", "sin", "tam"))

        english, sinhala, tamil, mixed = (run_pipeline(text=text) for text in texts)
        self.assertEqual(english.language_detection_status, "uncertain")
        self.assertEqual(
            (sinhala.detected_language, sinhala.language_detection_status),
            ("si", "detected"),
        )
        self.assertEqual(
            (tamil.detected_language, tamil.language_detection_status),
            ("ta", "detected"),
        )
        self.assertEqual(mixed.language_detection_status, "uncertain")

    def test_configured_statistical_identifier_is_used_for_english(self):
        class EnglishIdentifier:
            def identify(self, text):
                return LanguageIdentification("en", "fasttext:test", 0.9)

        result = run_pipeline(language_identifier=EnglishIdentifier())
        self.assertEqual(
            (result.detected_language, result.language_detection_status),
            ("en", "detected"),
        )

    def test_existing_non_extraction_outcomes_are_preserved(self):
        for status in ("multiple_claims", "ambiguous", "no_legal_assertion"):
            with self.subTest(status=status):
                result = run_pipeline(
                    claim_provider=StubClaimProvider(
                        status=status,
                        extracted_claim=None,
                    ),
                )
                self.assertEqual(result.extraction_status, status)
                self.assertIsNone(result.extracted_legal_claim)
                self.assertIsNone(result.error)

    def test_existing_extraction_provider_failure_is_preserved_safely(self):
        result = run_pipeline(
            claim_provider=StubClaimProvider(error="provider_unavailable"),
        )
        self.assertEqual(result.extraction_status, "failed")
        self.assertEqual(
            result.error.code,
            "claim_extraction_provider_unavailable",
        )
        self.assertEqual(result.error.stage, "claim_extraction")
        self.assertEqual(result.ocr_raw_text, "A legal claim.")
        self.assertEqual(result.cleaned_text, "A legal claim.")


class TesseractOCRProviderTests(TestCase):
    def setUp(self):
        from PIL import Image

        source = BytesIO()
        Image.new("L", (2, 2), 255).save(source, format="PNG")
        self.image = ProcessedImage(source.getvalue(), "PNG", 2, 2)
        self.provider = TesseractOCRProvider()

    def test_missing_tesseract_is_reported_safely(self):
        import pytesseract

        with (
            patch.object(
                pytesseract,
                "get_languages",
                side_effect=pytesseract.TesseractNotFoundError(),
            ),
            self.assertRaises(OCRProviderError) as raised,
        ):
            self.provider.extract(self.image, languages=("eng",))
        self.assertEqual(raised.exception.reason, "ocr_provider_unavailable")

    def test_missing_language_pack_is_reported_safely(self):
        import pytesseract

        with (
            patch.object(pytesseract, "get_languages", return_value=["eng"]),
            self.assertRaises(OCRProviderError) as raised,
        ):
            self.provider.extract(self.image, languages=("eng", "sin", "tam"))
        self.assertEqual(raised.exception.reason, "ocr_language_unavailable")

    def test_reading_order_and_confidence_are_derived_from_word_data(self):
        import pytesseract

        data = {
            "text": ["Section", "10", "applies"],
            "conf": ["95", "85", "90"],
            "block_num": [1, 1, 1],
            "par_num": [1, 1, 1],
            "line_num": [1, 1, 2],
        }
        with (
            patch.object(
                pytesseract,
                "get_languages",
                return_value=["eng", "sin", "tam"],
            ),
            patch.object(pytesseract, "image_to_data", return_value=data),
        ):
            result = self.provider.extract(
                self.image,
                languages=("eng", "sin", "tam"),
            )
        self.assertEqual(result.raw_text, "Section 10\napplies")
        self.assertAlmostEqual(result.confidence, 0.90)
        self.assertEqual(result.method, "tesseract")
        self.assertEqual(
            tuple((line.line_number, line.text) for line in result.line_confidences),
            ((1, "Section 10"), (2, "applies")),
        )
        self.assertAlmostEqual(result.line_confidences[0].confidence, 0.90)
        self.assertAlmostEqual(result.line_confidences[1].confidence, 0.90)

    def test_psm_is_configurable_and_defaults_to_six(self):
        self.assertEqual(TesseractOCRProvider().psm, 6)
        self.assertEqual(TesseractOCRProvider(psm=11).psm, 11)
        with self.assertRaises(ValueError):
            TesseractOCRProvider(psm=4)

    def test_configured_psm_is_passed_to_tesseract(self):
        import pytesseract

        data = {"text": [], "conf": []}
        with (
            patch.object(pytesseract, "get_languages", return_value=["eng"]),
            patch.object(
                pytesseract, "image_to_data", return_value=data
            ) as image_to_data,
        ):
            TesseractOCRProvider(psm=11).extract(self.image, languages=("eng",))
        self.assertEqual(image_to_data.call_args.kwargs["config"], "--psm 11")
