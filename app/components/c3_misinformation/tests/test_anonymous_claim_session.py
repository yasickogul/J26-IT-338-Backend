"""Offline tests for the standalone anonymous claim session helper."""

import hashlib
import re
import unittest
from unittest.mock import patch

from app.components.c3_misinformation import anonymous_claim_session as session


class AnonymousClaimSessionTests(unittest.TestCase):
    def test_cookie_names_and_generated_token_format(self):
        self.assertEqual(session.COOKIE_NAME, "compnet_claim_session")
        self.assertEqual(session.SECURE_COOKIE_NAME, "__Host-compnet_claim_session")
        token = session.new_session_token()
        self.assertEqual(len(token), 43)
        self.assertRegex(token, re.compile(r"\A[A-Za-z0-9_-]{43}\Z"))
        self.assertTrue(session.valid_session_token(token))

    def test_generation_uses_32_random_bytes_without_altering_output(self):
        synthetic_token = "A" * 43
        with patch.object(session.secrets, "token_urlsafe", return_value=synthetic_token) as generate:
            self.assertEqual(session.new_session_token(), synthetic_token)
        generate.assert_called_once_with(32)

    def test_valid_synthetic_tokens_and_invalid_boundaries(self):
        for token in ("A" * 43, "a" * 43, "0" * 43, "_-" * 21 + "A"):
            with self.subTest(token=token):
                self.assertTrue(session.valid_session_token(token))
        invalid = (
            None, 123, b"A" * 43, "", "A" * 42, "A" * 44,
            "A" * 42 + "=", "A" * 42 + "+", "A" * 42 + "/",
            "A" * 42 + " ", "A" * 42 + "\n", "A" * 42 + "é",
        )
        for token in invalid:
            with self.subTest(token=token):
                self.assertFalse(session.valid_session_token(token))

    def test_hash_uses_domain_separator_and_ascii_token(self):
        token = "A" * 43
        expected = hashlib.sha256(b"compnet-claim-session-v1:\x00" + token.encode("ascii")).hexdigest()
        digest = session.session_token_hash(token)
        self.assertEqual(digest, expected)
        self.assertEqual(len(digest), 64)
        self.assertNotEqual(digest, hashlib.sha256(token.encode("ascii")).hexdigest())
        self.assertNotEqual(digest, session.session_token_hash("B" * 43))
        self.assertEqual(digest, session.session_token_hash(token))

    def test_hash_rejects_invalid_tokens(self):
        for token in (None, 1, "A" * 42, "A" * 44, "A" * 42 + "=", "A" * 42 + "\n"):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, "^invalid anonymous session token$"):
                    session.session_token_hash(token)


if __name__ == "__main__":
    unittest.main()
