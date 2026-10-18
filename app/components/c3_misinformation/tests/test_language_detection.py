"""Offline behavior checks for conservative language detection."""

from dataclasses import asdict
from types import SimpleNamespace
import unittest

from app.components.c3_misinformation.pipeline.language_detection import (
    LanguageIdentification,
    LanguageIdentifierError,
    LanguageIdentifierUnavailableError,
    detect_language,
)


def identifier(label, *, score=0.8):
    return SimpleNamespace(identify=lambda text: LanguageIdentification(label, "fake-model", score))


class LanguageDetectionTests(unittest.TestCase):
    def test_native_sinhala_and_tamil_without_model(self):
        for text, label in (("කමල නම", "si"), ("தமிழ் மொழி", "ta")):
            with self.subTest(text=text):
                result = detect_language(text)
                self.assertEqual((result.label, result.status, result.reason),
                                 (label, "detected", "native_script"))
                self.assertEqual(result.evidence_languages, (label,))
                self.assertIsNone(result.model_name)

    def test_english_requires_identifier_and_multiple_tokens(self):
        self.assertEqual(detect_language("English words").reason, "model_unavailable")
        result = detect_language("English words", language_identifier=identifier("en"))
        self.assertEqual(asdict(result), {
            "label": "en", "status": "detected", "reason": "statistical_id",
            "evidence_languages": ("en",), "model_name": "fake-model", "model_score": 0.8,
        })
        self.assertEqual(detect_language("English", language_identifier=identifier("en")).reason,
                         "insufficient_language_evidence")
        self.assertEqual(detect_language("English words", language_identifier=identifier("si")).reason,
                         "romanized_language_uncertain")

    def test_mixed_native_scripts_and_insufficient_mix(self):
        result = detect_language("කමල නම தமிழ் மொழி")
        self.assertEqual((result.label, result.status, result.reason),
                         ("mixed", "detected", "mixed_language_evidence"))
        self.assertEqual(result.evidence_languages, ("si", "ta"))
        self.assertEqual(detect_language("කමල தமிழ்").reason,
                         "insufficient_mixed_language_evidence")

    def test_mixed_native_and_english_requires_segment_confirmation(self):
        def identify(text):
            label = "en" if text == "English words" else "mixed"
            return LanguageIdentification(label, "fake-model", 0.7)

        detector = SimpleNamespace(identify=identify)
        result = detect_language("කමල නම English words", language_identifier=detector)
        self.assertEqual((result.label, result.reason, result.evidence_languages),
                         ("mixed", "mixed_language_evidence", ("si", "en")))
        self.assertEqual(detect_language("කමල English words", language_identifier=detector).reason,
                         "insufficient_mixed_language_evidence")
        conflict = detect_language("කමල නම English words", language_identifier=identifier("si"))
        self.assertEqual((conflict.status, conflict.reason),
                         ("uncertain", "language_evidence_conflict"))

    def test_unsupported_uncertain_and_invalid_inputs(self):
        self.assertEqual(detect_language("  ").status, "invalid")
        with self.assertRaisesRegex(TypeError, "cleaned_text must be a string"):
            detect_language(None)
        self.assertEqual(detect_language("123 !").reason, "insufficient_language_evidence")
        self.assertEqual(detect_language("Αθήνα").reason, "unsupported_or_unrecognized_script")
        self.assertEqual(detect_language("කමල Αθήνα").evidence_languages, ("si",))
        unsupported = detect_language("Αθήνα", language_identifier=identifier("unsupported"))
        self.assertEqual((unsupported.status, unsupported.reason),
                         ("unsupported", "unsupported_language"))
        self.assertEqual(detect_language("English words", language_identifier=identifier("unsupported")).reason,
                         "unsupported_language_candidate")
        self.assertEqual(detect_language("English words", language_identifier=identifier("uncertain")).reason,
                         "insufficient_language_evidence")

    def test_identifier_errors_and_result_validation(self):
        for error, status, reason in (
            (LanguageIdentifierUnavailableError(), "uncertain", "model_unavailable"),
            (LanguageIdentifierError(), "detector_error", "model_error"),
        ):
            with self.subTest(error=error):
                def fail(text):
                    raise error
                result = detect_language("English words", language_identifier=SimpleNamespace(identify=fail))
                self.assertEqual((result.status, result.reason), (status, reason))
        for bad_result, error in (
            ("en", TypeError),
            (LanguageIdentification("wrong", "model"), ValueError),
            (LanguageIdentification("en", " "), ValueError),
            (LanguageIdentification("en", "model", float("nan")), ValueError),
        ):
            with self.subTest(bad_result=bad_result), self.assertRaises(error):
                detect_language("English words", language_identifier=SimpleNamespace(identify=lambda text: bad_result))


if __name__ == "__main__":
    unittest.main()
