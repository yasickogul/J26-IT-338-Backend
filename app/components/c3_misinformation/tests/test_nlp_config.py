"""Offline checks for shared C3 Phase 5 feature-flag parsing."""

import os
import unittest
from unittest.mock import patch

from app.components.c3_misinformation.config import (
    boolean_env,
    claim_extraction_enabled,
    language_detection_enabled,
)


class NLPConfigTests(unittest.TestCase):
    def test_flags_preserve_disabled_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(language_detection_enabled())
            self.assertFalse(claim_extraction_enabled())

    def test_supported_boolean_spellings_are_deterministic(self):
        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"C3_TEST_FLAG": value}, clear=True,
            ):
                self.assertTrue(boolean_env("C3_TEST_FLAG"))
        for value in ("0", "false", "NO", "off", ""):
            with self.subTest(value=value), patch.dict(
                os.environ, {"C3_TEST_FLAG": value}, clear=True,
            ):
                self.assertFalse(boolean_env("C3_TEST_FLAG", default=True))

    def test_invalid_configuration_fails_explicitly(self):
        with (
            patch.dict(os.environ, {"C3_TEST_FLAG": "perhaps"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "must be a boolean value"),
        ):
            boolean_env("C3_TEST_FLAG")
        with self.assertRaises(ValueError):
            boolean_env(" ")
        with self.assertRaises(TypeError):
            boolean_env("C3_TEST_FLAG", default=1)


if __name__ == "__main__":
    unittest.main()
