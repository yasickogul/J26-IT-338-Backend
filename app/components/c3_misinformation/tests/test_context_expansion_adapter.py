"""Offline source comparisons; no SQL, model imports, or database sessions."""
import ast
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
import unittest
from unittest.mock import AsyncMock

from app.components.c3_misinformation.pipeline import context_expansion as contracts
from app.components.c3_misinformation.pipeline.context_expansion_adapter import expand_parent_context
from app.components.c3_misinformation.pipeline.retrieval_contract import RetrievalResult


def ranked(cid="child", label="Part > Section", source="s"):
    return RetrievalResult(cid, cid, "child text", {
        "source_id": source, "section_label": label,
    }, None, None, 0.2, 0.8)


def row(cid="parent", label="Part", source="s"):
    return dict(chunk_id=cid, text="exact parent text", metadata={
        "source_id": source, "section_label": label,
    }, parent_context={"heading": "heading"}, paragraph=None)


def candidates(rows, source_id, section_label):
    return sorted(
        [r for r in rows if r["metadata"].get("source_id") == source_id
         and r["metadata"].get("section_label") == section_label],
        key=lambda r: r["chunk_id"],
    )


class SourceRows:
    """In-memory replacement for the source SQL result, never a DB session."""
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def execute(self, statement, params):
        self.calls.append(params)
        self.matches = candidates(self.rows, **params)
        return self

    def mappings(self):
        return self

    def all(self):
        return self.matches


class ContextExpansionAdapterTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        path = Path("/home/sano_01/fyp/backend/app/services/context_expansion.py")
        if not path.is_file():
            cls.original = None
            return
        # Execute only the original function, with inert SQL text and existing
        # contracts; do not import source database or retrieval infrastructure.
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "expand_parent_context")
        namespace = dict(vars(contracts))
        namespace.update(Sequence=Sequence, Any=Any, RetrievalResult=RetrievalResult,
                         Session=object, text=lambda sql: sql)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        cls.original = staticmethod(namespace["expand_parent_context"])

    async def compare(self, inputs, rows, **bounds):
        async def lookup(**params):
            return candidates(rows, **params)
        mock = AsyncMock(side_effect=lookup)
        before = [asdict(r) for r in inputs]
        actual = await expand_parent_context(inputs, parent_lookup=mock, **bounds)
        self.assertEqual(before, [asdict(r) for r in inputs])
        if self.original is not None:
            source = SourceRows(rows)
            expected = self.original(inputs, source, **bounds)
            self.assertEqual(asdict(actual), asdict(expected))
            self.assertEqual([c.kwargs for c in mock.await_args_list], source.calls)
        return actual, mock

    async def test_success_preserves_fields(self):
        result, lookup = await self.compare([ranked()], [row()])
        self.assertEqual(result.parents[0].text, "exact parent text")
        self.assertEqual(result.parents[0].relationship, "structural_parent_of")
        self.assertEqual(result.parents[0].depth, 1)
        lookup.assert_awaited_once_with(source_id="s", section_label="Part")

    async def test_missing_parent_notice(self):
        result, _ = await self.compare([ranked()], [])
        self.assertEqual(result.unresolved[0].reason,
                         "no same-source passage has the parent section label")

    async def test_ambiguity_does_not_choose_first_candidate(self):
        result, _ = await self.compare([ranked()], [row("z"), row("a")])
        self.assertEqual(result.parents, ())
        self.assertEqual(result.unresolved[0].reason,
                         "ambiguous parent: 2 same-source passages matched")

    async def test_nested_breadth_first_order_and_depth(self):
        rows = [row("a", "A > B"), row("root", "A"), row("z", "Z")]
        result, _ = await self.compare(
            [ranked("c1", "A > B > C"), ranked("c2", "Z > C")],
            rows, max_depth=3,
        )
        self.assertEqual([p.parent_chunk_id for p in result.parents], ["a", "z", "root"])
        self.assertEqual([p.depth for p in result.parents], [1, 1, 2])
        self.assertEqual(result.unresolved[-1].reason, "no hierarchical parent label")

    async def test_lookup_contract_excludes_other_source_and_wrong_label(self):
        result, lookup = await self.compare([ranked()], [
            row("foreign", source="other"), row("wrong", label="Other"), row(),
        ])
        self.assertEqual([p.parent_chunk_id for p in result.parents], ["parent"])
        lookup.assert_awaited_once_with(source_id="s", section_label="Part")

    async def test_source_values_are_not_trimmed(self):
        result, lookup = await self.compare([ranked(source=" s ")], [row(source=" s ")])
        self.assertEqual(result.parents[0].source_id, " s ")
        lookup.assert_awaited_once_with(source_id=" s ", section_label="Part")

    async def test_duplicate_parent_keeps_first_child_only(self):
        result, lookup = await self.compare([ranked("c1"), ranked("c2")], [row()])
        self.assertEqual(len(result.parents), 1)
        self.assertEqual(result.parents[0].child_chunk_id, "c1")
        self.assertEqual(lookup.await_count, 2)

    async def test_default_depth_stops_after_one(self):
        result, lookup = await self.compare([ranked(label="A > B > C")],
                                            [row("p", "A > B"), row("root", "A")])
        self.assertEqual(len(result.parents), 1)
        self.assertEqual(lookup.await_count, 1)

    async def test_passage_cap_preserves_original_notice_label(self):
        result, lookup = await self.compare(
            [ranked("c1"), ranked("c2", "Other > Child")], [row()],
            max_passages=1,
        )
        self.assertEqual(lookup.await_count, 1)
        notice = result.unresolved[0]
        self.assertEqual(notice.reason, "parent expansion limit reached")
        self.assertEqual(notice.requested_parent_section_label, "Other > Child")
        self.assertEqual(notice.child_chunk_id, "c2")

    async def test_empty_results(self):
        result, lookup = await self.compare([], [])
        self.assertEqual(result, contracts.ContextExpansion((), ()))
        lookup.assert_not_awaited()

    async def test_invalid_hierarchy_and_source_notices(self):
        result, lookup = await self.compare([
            ranked("plain", "Part"), ranked("missing", "Part > Section", None),
            ranked("blank", "Part > Section", " "),
            ranked("invalid", None, 42),
        ], [])
        self.assertEqual([n.reason for n in result.unresolved], [
            "no hierarchical parent label", "missing source_id",
            "missing source_id", "no hierarchical parent label",
        ])
        lookup.assert_not_awaited()

    async def test_invalid_bounds_before_lookup(self):
        for bounds in (dict(max_depth=0), dict(max_depth=4), dict(max_depth=True),
                       dict(max_passages=0), dict(max_passages=11), dict(max_passages=1.0)):
            with self.subTest(bounds=bounds):
                lookup = AsyncMock()
                with self.assertRaises(ValueError) as caught:
                    await expand_parent_context([], parent_lookup=lookup, **bounds)
                lookup.assert_not_awaited()
                if self.original is not None:
                    with self.assertRaises(ValueError) as original:
                        self.original([], SourceRows([]), **bounds)
                    self.assertEqual(str(caught.exception), str(original.exception))

    async def test_lookup_error_propagates(self):
        error = RuntimeError("lookup unavailable")
        with self.assertRaises(RuntimeError) as caught:
            await expand_parent_context([ranked()], parent_lookup=AsyncMock(side_effect=error))
        self.assertIs(caught.exception, error)

    async def test_missing_candidate_field_preserves_key_error(self):
        candidate = row()
        del candidate["paragraph"]
        with self.assertRaises(KeyError) as caught:
            await expand_parent_context([ranked()], parent_lookup=AsyncMock(return_value=[candidate]))
        self.assertEqual(caught.exception.args, ("paragraph",))
        if self.original is not None:
            with self.assertRaises(KeyError) as original:
                self.original([ranked()], SourceRows([candidate]))
            self.assertEqual(original.exception.args, caught.exception.args)


if __name__ == "__main__":
    unittest.main()
