"""Offline tests for the local-only LaBSE token admission rule."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from app.components.c3_misinformation.pipeline import claim_input_tokens as tokens


class ClaimInputTokenTests(unittest.TestCase):
    def tearDown(self):
        tokens._local_tokenizer_and_limit.cache_clear()

    def test_counts_include_special_tokens_without_truncation(self):
        tokenizer = Mock(return_value={"input_ids": [[1, 2, 3], [1, 2, 3, 4]]})
        with patch.object(tokens, "_local_tokenizer_and_limit", return_value=(tokenizer, 4)):
            self.assertEqual(tokens.validate_claim_input_tokens(["first", "second"]), [3, 4])
            self.assertEqual(tokens.effective_claim_token_limit(), 4)
        tokenizer.assert_called_once_with(
            ["first", "second"], truncation=False, add_special_tokens=True, verbose=False,
        )

    def test_over_limit_reports_first_index_count_and_limit(self):
        tokenizer = Mock(return_value={"input_ids": [[1, 2], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]})
        with (
            patch.object(tokens, "_local_tokenizer_and_limit", return_value=(tokenizer, 4)),
            self.assertRaises(tokens.EmbeddingInputTooLongError) as caught,
        ):
            tokens.validate_claim_input_tokens(["first", "second", "third"])
        self.assertEqual((caught.exception.index, caught.exception.token_count, caught.exception.limit),
                         (1, 5, 4))
        self.assertEqual(str(caught.exception),
                         "Input[1] has 5 tokens (limit 4); truncation is not allowed.")

    def test_invalid_texts_fail_before_tokenizer_access(self):
        invalid = [
            ("text", TypeError), (b"text", TypeError), (None, TypeError),
            ([], ValueError), (["ok", None], TypeError), (["ok", " \n "], ValueError),
        ]
        with patch.object(tokens, "_local_tokenizer_and_limit") as loader:
            for value, error in invalid:
                with self.subTest(value=value), self.assertRaises(error):
                    tokens.validate_claim_input_tokens(value)
            loader.assert_not_called()

    def test_tokenizer_call_failure_is_wrapped(self):
        tokenizer = Mock(side_effect=RuntimeError("private tokenizer detail"))
        with (
            patch.object(tokens, "_local_tokenizer_and_limit", return_value=(tokenizer, 256)),
            self.assertRaises(tokens.TokenizerUnavailableError) as caught,
        ):
            tokens.validate_claim_input_tokens(["claim"])
        self.assertEqual(str(caught.exception), "Local claim tokenizer is unavailable.")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_cached_resolution_is_local_fast_tokenizer_only(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "tokenizer.json").write_text("{}")
            (snapshot / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 256}))
            tokenizer = SimpleNamespace(is_fast=True)
            download = Mock(return_value=directory)
            from_pretrained = Mock(return_value=tokenizer)
            hub = ModuleType("huggingface_hub")
            hub.snapshot_download = download
            transformers = ModuleType("transformers")
            transformers.AutoTokenizer = SimpleNamespace(from_pretrained=from_pretrained)
            with patch.dict(sys.modules, {"huggingface_hub": hub, "transformers": transformers}):
                self.assertEqual(tokens._local_tokenizer_and_limit(), (tokenizer, 256))
                self.assertEqual(tokens.effective_claim_token_limit(), 256)
            download.assert_called_once_with(tokens.MODEL_NAME, local_files_only=True, allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "sentence_bert_config.json"])
            from_pretrained.assert_called_once_with(directory, local_files_only=True, use_fast=True)

    def test_missing_assets_and_invalid_cached_limit_are_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            download = Mock(return_value=directory)
            hub = ModuleType("huggingface_hub")
            hub.snapshot_download = download
            transformers = ModuleType("transformers")
            transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(is_fast=True)))
            with patch.dict(sys.modules, {"huggingface_hub": hub, "transformers": transformers}):
                with self.assertRaisesRegex(tokens.TokenizerUnavailableError, "Local claim tokenizer is unavailable"):
                    tokens._local_tokenizer_and_limit()
                (snapshot / "tokenizer.json").write_text("{}")
                for limit in (None, True, 2, "256"):
                    with self.subTest(limit=limit):
                        (snapshot / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": limit}))
                        with self.assertRaises(tokens.TokenizerUnavailableError):
                            tokens._local_tokenizer_and_limit()

    def test_missing_dependency_is_reported_as_unavailable(self):
        with (
            patch.dict(sys.modules, {"huggingface_hub": None}),
            self.assertRaises(tokens.TokenizerUnavailableError) as caught,
        ):
            tokens._local_tokenizer_and_limit()
        self.assertEqual(str(caught.exception), "Local claim tokenizer is unavailable.")
        self.assertIsInstance(caught.exception.__cause__, ImportError)


if __name__ == "__main__":
    unittest.main()
