"""FastText adapter tests using fake local models only."""

import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.components.c3_misinformation.pipeline.fasttext_language_identifier import (
    FastTextLanguageIdentifier,
    MODEL_PATH_ENV,
)
from app.components.c3_misinformation.pipeline.language_detection import (
    LanguageIdentifierError,
    LanguageIdentifierUnavailableError,
)


class FastTextAdapterTests(unittest.TestCase):
    def test_missing_model_is_lazy_and_unavailable(self):
        with patch.dict(os.environ, {MODEL_PATH_ENV: ""}):
            adapter = FastTextLanguageIdentifier()
            with self.assertRaises(LanguageIdentifierUnavailableError):
                adapter.identify("English words")
        with self.assertRaisesRegex(LanguageIdentifierError, "empty text"):
            adapter.identify("  ")
        with self.assertRaisesRegex(TypeError, "text must be a string"):
            adapter.identify(None)

    def test_fake_model_prediction_and_loader_cache(self):
        with tempfile.NamedTemporaryFile() as file:
            model = SimpleNamespace(predict=Mock(return_value=(["__label__en"], [0.91])))
            loader = Mock(return_value=model)
            adapter = FastTextLanguageIdentifier(file.name, model_loader=loader)
            self.assertEqual(adapter.identify("English\nwords").label, "en")
            self.assertEqual(adapter.identify("More English").score, 0.91)
            loader.assert_called_once_with(file.name)
            model.predict.assert_any_call("English words", k=1)

    def test_unsupported_label_and_malformed_predictions(self):
        with tempfile.NamedTemporaryFile() as file:
            model = SimpleNamespace(predict=Mock(return_value=(["__label__fr"], [0.8])))
            adapter = FastTextLanguageIdentifier(file.name, model_loader=lambda path: model,
                                                 model_name="fake-fasttext")
            result = adapter.identify("French words")
            self.assertEqual((result.label, result.model_name, result.score),
                             ("unsupported", "fake-fasttext", 0.8))
            for output in (([], []), (["__label__en"], [float("nan")])):
                with self.subTest(output=output), self.assertRaises(LanguageIdentifierError):
                    model.predict.return_value = output
                    adapter.identify("English words")


if __name__ == "__main__":
    unittest.main()
