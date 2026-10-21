"""Offline contract tests for the C3 image extraction endpoint."""

from unittest import TestCase

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.components.c3_misinformation.image_api import (
    get_image_claim_extractor_dependency,
)
from app.components.c3_misinformation.image_claim_service import (
    ImageClaimExtractionResult,
)
from app.components.c3_misinformation.pipeline.ocr_extractor import (
    OCRLineConfidence,
)
from app.components.c3_misinformation.router import router


class ImageApiTests(TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(router, prefix="/api/v1/c3")
        self.seen = []

        def extractor(image_bytes):
            self.seen.append(image_bytes)
            return ImageClaimExtractionResult(
                ocr_raw_text="Section 10 applies.",
                ocr_status="completed",
                cleaned_text="Section 10 applies.",
                extracted_legal_claim="Section 10 applies.",
                ocr_confidence=0.91,
                ocr_languages=("eng", "sin", "tam"),
                low_confidence=False,
                extraction_status="extracted",
                detected_language=None,
                language_detection_status="uncertain",
                language_detection_reason="model_unavailable",
                ocr_method="stub-ocr",
                claim_extraction_method="stub-claim",
                ocr_line_confidences=(
                    OCRLineConfidence(1, "Section 10 applies.", 0.91),
                ),
            )

        self.app.dependency_overrides[get_image_claim_extractor_dependency] = lambda: extractor
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.dependency_overrides.clear()

    def test_multipart_image_endpoint_returns_structured_extraction(self):
        response = self.client.post(
            "/api/v1/c3/claims/extract-image",
            files={"image": ("claim.png", b"png-content", "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.seen, [b"png-content"])
        self.assertEqual(
            response.json(),
            {
                "ocr_raw_text": "Section 10 applies.",
                "ocr_status": "completed",
                "cleaned_text": "Section 10 applies.",
                "extracted_legal_claim": "Section 10 applies.",
                "ocr_confidence": 0.91,
                "ocr_languages": ["eng", "sin", "tam"],
                "low_confidence": False,
                "extraction_status": "extracted",
                "detected_language": None,
                "language_detection_status": "uncertain",
                "language_detection_reason": "model_unavailable",
                "ocr_method": "stub-ocr",
                "claim_extraction_method": "stub-claim",
                "ocr_line_confidences": [
                    {
                        "line_number": 1,
                        "text": "Section 10 applies.",
                        "confidence": 0.91,
                    }
                ],
                "error": None,
            },
        )

    def test_missing_multipart_image_is_rejected(self):
        response = self.client.post("/api/v1/c3/claims/extract-image")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.seen, [])

    def test_invalid_image_media_type_is_rejected_before_extraction(self):
        response = self.client.post(
            "/api/v1/c3/claims/extract-image",
            files={"image": ("claim.txt", b"not-an-image", "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.seen, [])
        self.assertEqual(response.json()["extraction_status"], "failed")
        self.assertEqual(
            response.json()["error"],
            {
                "code": "unsupported_image_type",
                "stage": "image_validation",
            },
        )
        self.assertEqual(response.json()["language_detection_status"], "not_run")

    def test_endpoint_is_registered_in_full_application_schema(self):
        from app.main import app

        self.assertIn(
            "/api/v1/c3/claims/extract-image",
            app.openapi()["paths"],
        )
