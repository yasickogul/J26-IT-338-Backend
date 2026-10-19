"""Focused mocked tests for the standalone legal-claim translation contract."""

import asyncio
import unittest

from app.components.c3_misinformation.pipeline.claim_translation import (
    ClaimTranslationResult,
    TranslationProviderError,
    TranslationProviderResponse,
    translate_claim,
)


class MockTranslationProvider:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def translate(self, text, *, source_language, target_language):
        self.calls.append((text, source_language, target_language))
        if self.error is not None:
            raise self.error
        return self.response


class ClaimTranslationTests(unittest.TestCase):
    def test_english_is_skipped_without_calling_provider(self):
        provider = MockTranslationProvider()
        result = translate_claim("The court applies the law.", "en", provider=provider)

        self.assertEqual(result, ClaimTranslationResult(
            original_text="The court applies the law.",
            source_language="en",
            target_language="en",
            translated_text=None,
            method=None,
            status="skipped",
        ))
        self.assertEqual(provider.calls, [])

    def test_sinhala_translation_uses_injected_provider(self):
        original = "මෙම නීතිය අදාළ වේ."
        provider = MockTranslationProvider(
            TranslationProviderResponse(
                "This law applies.", "mock:si-en:test-v1",
            ),
        )

        result = translate_claim(original, "si", provider=provider)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.original_text, original)
        self.assertEqual(result.source_language, "si")
        self.assertEqual(result.target_language, "en")
        self.assertEqual(result.translated_text, "This law applies.")
        self.assertEqual(result.method, "mock:si-en:test-v1")
        self.assertEqual(provider.calls, [(original, "si", "en")])

    def test_tamil_translation_uses_injected_provider(self):
        original = "இந்த சட்டம் பொருந்தும்."
        provider = MockTranslationProvider(
            TranslationProviderResponse(
                "This law applies.", "mock:ta-en:test-v1",
            ),
        )

        result = translate_claim(original, "ta", provider=provider)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.original_text, original)
        self.assertEqual(result.translated_text, "This law applies.")
        self.assertEqual(provider.calls, [(original, "ta", "en")])

    def test_missing_uncertain_and_unsupported_language_are_not_translated(self):
        provider = MockTranslationProvider(
            TranslationProviderResponse("fabricated", "must-not-run"),
        )
        for source_language in (None, "uncertain", "unsupported"):
            with self.subTest(source_language=source_language):
                result = translate_claim(
                    "Claim text", source_language, provider=provider,
                )
                self.assertEqual(result.status, "uncertain")
                self.assertIsNone(result.translated_text)
                self.assertEqual(result.source_language, source_language)
        self.assertEqual(provider.calls, [])

    def test_mixed_claim_is_not_sent_as_a_whole_to_provider(self):
        provider = MockTranslationProvider(
            TranslationProviderResponse("fabricated", "must-not-run"),
        )
        result = translate_claim(
            "මෙම law applies.", "mixed", provider=provider,
        )
        self.assertEqual(result.status, "uncertain")
        self.assertIsNone(result.translated_text)
        self.assertEqual(provider.calls, [])

    def test_missing_provider_fails_without_fabricating_translation(self):
        result = translate_claim("මෙම නීතිය අදාළ වේ.", "si")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "provider_unavailable")
        self.assertIsNone(result.translated_text)

    def test_provider_failure_returns_safe_code_without_sensitive_exception_text(self):
        original = "Sensitive legal claim content"
        provider = MockTranslationProvider(
            error=TranslationProviderError("provider_error"),
        )
        result = translate_claim(original, "ta", provider=provider)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "provider_error")
        self.assertIsNone(result.translated_text)
        self.assertEqual(result.original_text, original)
        self.assertNotIn(original, str(result.error_code))

    def test_legal_text_is_passed_verbatim_and_original_is_preserved(self):
        original = (
            "No person shall be evicted unless notice is given under "
            "section 31B, except where a court orders otherwise."
        )
        translated = (
            "No person shall be evicted unless notice is given under "
            "section 31B, except where a court orders otherwise."
        )
        provider = MockTranslationProvider(
            TranslationProviderResponse(translated, "mock:legal-fidelity-fixture"),
        )

        result = translate_claim(original, "si", provider=provider)

        self.assertEqual(provider.calls[0][0], original)
        self.assertEqual(result.original_text, original)
        self.assertEqual(result.translated_text, translated)
        for legal_detail in ("No person", "unless", "section 31B", "except", "otherwise"):
            self.assertIn(legal_detail, result.original_text)
            self.assertIn(legal_detail, result.translated_text)

    def test_invalid_provider_response_is_failed_and_not_returned(self):
        provider = MockTranslationProvider(
            TranslationProviderResponse("  ", "mock:empty"),
        )
        result = translate_claim("தமிழ் உரை.", "ta", provider=provider)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "invalid_provider_response")
        self.assertIsNone(result.translated_text)

    def test_empty_input_fails_before_provider_call(self):
        provider = MockTranslationProvider(
            TranslationProviderResponse("should not run", "mock"),
        )
        result = translate_claim(" \n ", "si", provider=provider)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "invalid_input")
        self.assertIsNone(result.translated_text)
        self.assertEqual(provider.calls, [])

    def test_invalid_input_types_and_language_labels(self):
        with self.assertRaisesRegex(TypeError, "original_text must be a string"):
            translate_claim(None, "en")
        for language in ("fr", "SI", "", 1):
            with self.subTest(language=language):
                with self.assertRaisesRegex(ValueError, "source_language is not supported"):
                    translate_claim("Claim text", language)

    def test_invalid_provider_response_forms(self):
        for response in (
            None,
            {"translated_text": "English", "method": "mock"},
            TranslationProviderResponse("English", " "),
            TranslationProviderResponse(123, "mock"),
        ):
            with self.subTest(response=response):
                result = translate_claim(
                    "தமிழ் உரை.", "ta", provider=MockTranslationProvider(response),
                )
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.error_code, "invalid_provider_response")
                self.assertIsNone(result.translated_text)

    def test_provider_error_codes_and_unexpected_exception(self):
        for code in ("provider_unavailable", "provider_error", "invalid_provider_response"):
            with self.subTest(code=code):
                result = translate_claim(
                    "தமிழ் உரை.", "ta",
                    provider=MockTranslationProvider(error=TranslationProviderError(code)),
                )
                self.assertEqual((result.status, result.error_code), ("failed", code))
        with self.assertRaisesRegex(ValueError, "provider error code is invalid"):
            TranslationProviderError("invalid_input")
        with self.assertRaises(RuntimeError):
            translate_claim(
                "தமிழ் உரை.", "ta", provider=MockTranslationProvider(error=RuntimeError("unexpected")),
            )

    def test_provider_cancellation_propagates(self):
        error = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError) as caught:
            translate_claim(
                "தமிழ் உரை.", "ta", provider=MockTranslationProvider(error=error),
            )
        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()
