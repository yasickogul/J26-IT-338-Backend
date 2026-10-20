"""Offline tests for CompNet's independent retrieval contract."""

from dataclasses import FrozenInstanceError, MISSING, asdict, fields
from typing import Any, get_type_hints
import unittest

from app.components.c3_misinformation.pipeline import retrieval_contract as contract


class RetrievalContractTests(unittest.TestCase):
    def test_top_k_valid_boundaries(self):
        self.assertEqual(contract.MAX_TOP_K, 50)
        for value in (1, 5, 50):
            with self.subTest(value=value):
                self.assertIsNone(contract._validate_top_k(value))

    def test_top_k_rejects_invalid_types(self):
        for value in (True, False, 1.0, "1", None, [], 1 + 0j):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, r"^top_k must be a positive integer$"):
                    contract._validate_top_k(value)

    def test_top_k_rejects_out_of_range_integers(self):
        for value in (-1, 0, 51, 100):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"^top_k must be between 1 and 50$"):
                    contract._validate_top_k(value)

    def test_filter_accepts_none_and_trims_strings(self):
        for name in ("source_id", "legal_domain", "language"):
            with self.subTest(name=name):
                self.assertIsNone(contract._validate_filter(name, None))
                self.assertEqual(contract._validate_filter(name, "  English\t"), "English")
                self.assertEqual(contract._validate_filter(name, "தமிழ்"), "தமிழ்")

    def test_filter_rejects_blank_and_nonstring_values(self):
        for value in ("", " \t\n", 0, False, [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"^source_id must be a nonblank string when provided$"):
                    contract._validate_filter("source_id", value)
        with self.assertRaisesRegex(ValueError, r"^legal_domain must be a nonblank string when provided$"):
            contract._validate_filter("legal_domain", "  ")

    def test_result_preserves_fields_scores_and_metadata(self):
        result = contract.RetrievalResult(
            chunk_id="chunk-1", original_chunk_id="original-1", text="Legal text",
            metadata={"source_id": "act-1", "section_label": "Part > Section"},
            parent_context={"heading": "Part"}, paragraph="2",
            distance=0.2, similarity=0.8,
        )
        self.assertEqual(asdict(result), {
            "chunk_id": "chunk-1", "original_chunk_id": "original-1", "text": "Legal text",
            "metadata": {"source_id": "act-1", "section_label": "Part > Section"},
            "parent_context": {"heading": "Part"}, "paragraph": "2",
            "distance": 0.2, "similarity": 0.8,
        })
        self.assertEqual([field.name for field in fields(result)], [
            "chunk_id", "original_chunk_id", "text", "metadata", "parent_context",
            "paragraph", "distance", "similarity",
        ])
        self.assertTrue(all(field.default is MISSING and field.default_factory is MISSING for field in fields(result)))
        self.assertEqual(get_type_hints(contract.RetrievalResult)["metadata"], dict[str, Any])
        self.assertEqual(get_type_hints(contract.RetrievalResult)["parent_context"], dict[str, Any] | None)
        with self.assertRaises(FrozenInstanceError):
            result.distance = 0.1

    def test_result_allows_empty_text_and_optional_fields_as_source_does(self):
        result = contract.RetrievalResult("c", "o", "", {}, None, None, 0.0, 1.0)
        self.assertEqual(result.text, "")
        self.assertIsNone(result.parent_context)
        self.assertIsNone(result.paragraph)

    def test_database_backed_retrieval_is_absent(self):
        self.assertFalse(hasattr(contract, "retrieve"))


if __name__ == "__main__":
    unittest.main()
