"""Offline tests for independent context expansion definitions."""

from dataclasses import FrozenInstanceError, asdict, fields
from typing import Any, get_type_hints
import unittest

from app.components.c3_misinformation.pipeline import context_expansion as context


class ContextExpansionTests(unittest.TestCase):
    def test_parent_label_preserves_section_hierarchy(self):
        cases = {
            "Part > Section > Subsection": "Part > Section",
            "Part > Section": "Part",
            "Part > Section > Subsection > Paragraph": "Part > Section > Subsection",
            "Part > Section > ": "Part > Section",
            " > Section": "",
            " > ": "",
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assertEqual(context._parent_label(label), expected)

    def test_parent_label_rejects_unsupported_or_unstructured_input(self):
        for label in (None, "", "Section", "Part>Section", "Part >Section", "Part> Section", 123, ["Part", "Section"]):
            with self.subTest(label=label):
                self.assertIsNone(context._parent_label(label))

    def test_valid_bounds_include_both_endpoints(self):
        self.assertIsNone(context._validate_bounds(1, 1))
        self.assertIsNone(context._validate_bounds(3, 10))
        self.assertIsNone(context._validate_bounds(2, 5))
        self.assertEqual((context.MAX_PARENT_DEPTH, context.MAX_PARENT_PASSAGES), (3, 10))

    def test_invalid_depth_preserves_error(self):
        for depth in (0, 4, -1, True, False, 1.0, "1", None):
            with self.subTest(depth=depth):
                with self.assertRaisesRegex(ValueError, r"^max_depth must be between 1 and 3$"):
                    context._validate_bounds(depth, 1)

    def test_invalid_passage_count_preserves_error(self):
        for passages in (0, 11, -1, True, False, 1.0, "1", None):
            with self.subTest(passages=passages):
                with self.assertRaisesRegex(ValueError, r"^max_passages must be between 1 and 10$"):
                    context._validate_bounds(1, passages)

    def test_depth_validation_precedes_passage_validation(self):
        with self.assertRaisesRegex(ValueError, r"^max_depth must be between 1 and 3$"):
            context._validate_bounds(0, 0)

    def test_dataclasses_preserve_fields_types_and_frozen_contract(self):
        parent = context.ParentEvidence(
            parent_chunk_id="parent", child_chunk_id="child", source_id="act-1",
            relationship="structural_parent_of", depth=1, text="Parent passage",
            metadata={"section_label": "Part"}, parent_context={"heading": "Part"}, paragraph="1",
        )
        unresolved = context.UnresolvedParent(
            child_chunk_id="child", source_id=None,
            requested_parent_section_label="Part", depth=2, reason="missing source_id",
        )
        expansion = context.ContextExpansion((parent,), (unresolved,))
        self.assertEqual(expansion.parents, (parent,))
        self.assertEqual(expansion.unresolved, (unresolved,))
        self.assertEqual(asdict(parent)["metadata"], {"section_label": "Part"})
        self.assertIsNone(asdict(unresolved)["source_id"])
        self.assertEqual([field.name for field in fields(parent)], [
            "parent_chunk_id", "child_chunk_id", "source_id", "relationship", "depth",
            "text", "metadata", "parent_context", "paragraph",
        ])
        self.assertTrue(all(field.default is field.default_factory for field in fields(parent)))
        self.assertEqual(get_type_hints(context.ParentEvidence)["metadata"], dict[str, Any])
        self.assertEqual(get_type_hints(context.ParentEvidence)["parent_context"], dict[str, Any] | None)
        self.assertEqual(get_type_hints(context.ContextExpansion)["parents"], tuple[context.ParentEvidence, ...])
        with self.assertRaises(FrozenInstanceError):
            parent.depth = 2

    def test_empty_expansion_and_optional_fields(self):
        self.assertEqual(asdict(context.ContextExpansion((), ())), {"parents": (), "unresolved": ()})
        parent = context.ParentEvidence("p", "c", "s", "structural_parent_of", 1, "text", {}, None, None)
        self.assertIsNone(parent.parent_context)
        self.assertIsNone(parent.paragraph)

    def test_database_backed_entry_point_is_absent(self):
        self.assertFalse(hasattr(context, "expand_parent_context"))


if __name__ == "__main__":
    unittest.main()
