"""Offline behavior checks for the migrated text normalizer."""

import unittest

from app.components.c3_misinformation.pipeline.text_preprocessing import clean_text


class CleanTextTests(unittest.TestCase):
    def test_unicode_nfc_and_leading_bom(self):
        self.assertEqual(clean_text("\ufeffCafe\u0301\n\ufeffinside"), "Café\n\ufeffinside")

    def test_whitespace_line_endings_and_empty_input(self):
        cases = [
            ("", ""),
            (" \t\r\n\f \n", ""),
            ("\n \t First  line \r\n\r\n \nSecond\tline \r", "First line\n\nSecond line"),
            ("One\rTwo\r\nThree", "One\nTwo\nThree"),
            (" A\n\n\nB ", "A\n\nB"),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(clean_text(source), expected)

    def test_page_boundary_duplicate_removal_is_opt_in(self):
        source = "Header\nRepeated\fRepeated\nBody"
        self.assertEqual(clean_text(source), "Header\nRepeated\nRepeated\nBody")
        self.assertEqual(
            clean_text(source, remove_page_boundary_duplicates=True),
            "Header\nRepeated\nBody",
        )

    def test_only_identical_adjacent_page_boundary_lines_are_removed(self):
        cases = [
            ("A\fB", "A\nB"),
            ("A\f\fA", "A\nA"),
            ("A\nA", "A\nA"),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(
                    clean_text(source, remove_page_boundary_duplicates=True), expected
                )

    def test_text_must_be_a_string(self):
        for value in (None, 123, b"text"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "^text must be a string$"):
                    clean_text(value)

    def test_duplicate_removal_flag_must_be_boolean(self):
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TypeError, "^remove_page_boundary_duplicates must be a boolean$"
                ):
                    clean_text("A", remove_page_boundary_duplicates=value)


if __name__ == "__main__":
    unittest.main()
