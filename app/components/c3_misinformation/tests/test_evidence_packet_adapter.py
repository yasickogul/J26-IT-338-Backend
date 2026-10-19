"""Offline async orchestration and isolated original-builder comparisons."""
import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from functools import partial
import unittest
from unittest.mock import AsyncMock, Mock, call

from app.components.c3_misinformation.pipeline.evidence_packet_adapter import build_evidence_packet
from app.components.c3_misinformation.pipeline.context_expansion import (
    ContextExpansion, ParentEvidence, UnresolvedParent,
)
from app.components.c3_misinformation.pipeline.retrieval_contract import RetrievalResult
from app.components.c3_misinformation.pipeline.context_expansion_adapter import expand_parent_context
from app.components.c3_misinformation.pipeline.retrieval_adapter import retrieve


def metadata(label):
    return {"source_id": "act", "section_label": label, "title": "Title",
            "language": "English", "legal_domain": "Civil", "act_number": "1",
            "source_file": "act.pdf", "document_type": "act"}


def child(cid="child", label="Part > Section"):
    return RetrievalResult(cid, "original-" + cid, "Complete தமிழ் text\n",
                           metadata(label), None, "1", 0.2, 0.8)


def parent(cid="parent", child_id="child", label="Part"):
    return ParentEvidence(cid, child_id, "act", "structural_parent_of", 1,
                          "Complete parent text", metadata(label), None, None)


class EvidencePacketAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ranked = [child()]
        self.expansion = ContextExpansion((), ())
        self.retrieve = AsyncMock(return_value=self.ranked)
        self.expand = AsyncMock(side_effect=lambda *a, **k: self.expansion)
        self.lookup = AsyncMock(side_effect=lambda cid: "original-" + cid)

    async def build(self, query=" query ", **kwargs):
        return await build_evidence_packet(
            query, retrieve=self.retrieve, expand_parent_context=self.expand,
            original_chunk_lookup=self.lookup, **kwargs,
        )

    def original(self, query=" query ", **kwargs):
        path = Path("/home/sano_01/fyp/backend/app/services/evidence_packet.py")
        if not path.is_file():
            self.skipTest("Original CompNet source unavailable for comparison")
        retrieval = Mock(return_value=self.ranked)
        expansion = Mock(return_value=self.expansion)
        # Only replace this isolated module's imports; no sys.modules changes,
        # source dependency imports, SQL execution, or database connection.
        modules = {
            "sqlalchemy": SimpleNamespace(text=lambda sql: sql),
            "app.services.retrieval": SimpleNamespace(retrieve=retrieval),
            "app.services.context_expansion": SimpleNamespace(expand_parent_context=expansion),
        }
        def isolated_import(name, *args, **kwargs):
            return modules[name] if name in modules else builtins.__import__(name, *args, **kwargs)
        namespace = {"__builtins__": dict(vars(builtins), __import__=isolated_import)}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        db = Mock()
        db.execute.side_effect = lambda statement, params: SimpleNamespace(
            scalar_one=lambda: "original-" + params["chunk_id"])
        result = namespace["build_evidence_packet"](query, db, **kwargs)
        self.assertEqual([c.args[1]["chunk_id"] for c in db.execute.call_args_list],
                         [c.args[0] for c in self.lookup.await_args_list])
        self.assertEqual(retrieval.call_args.kwargs, self.retrieve.await_args.kwargs)
        self.assertEqual(expansion.call_args.kwargs, self.expand.await_args.kwargs)
        return result

    async def test_ranked_without_parents_matches_original(self):
        self.ranked.append(child("second"))
        packet = await self.build()
        self.assertEqual(packet, self.original())
        self.assertEqual([r["evidence_id"] for r in packet["ranked_evidence"]],
                         ["chunk:child", "chunk:second"])
        self.assertEqual(packet["evidence"][0]["text"], self.ranked[0].text)
        self.lookup.assert_not_awaited()
        self.assertEqual(json.loads(json.dumps(packet)), packet)

    async def test_resolved_parent_and_exact_identifiers(self):
        self.expansion = ContextExpansion((parent(),), ())
        packet = await self.build()
        self.assertEqual(packet, self.original())
        self.assertEqual(packet["context_evidence_ids"], ["chunk:parent"])
        self.assertEqual(packet["evidence"][-1]["original_chunk_id"], "original-parent")
        self.assertEqual(packet["relationships"][0]["child_evidence_id"], "chunk:child")
        self.lookup.assert_awaited_once_with("parent")

    async def test_multiple_parents_and_lookup_order(self):
        self.ranked.append(child("second", "Other > Child"))
        self.expansion = ContextExpansion((parent(), parent("other", "second", "Other")), ())
        packet = await self.build()
        self.assertEqual(packet, self.original())
        self.assertEqual(self.lookup.await_args_list, [call("parent"), call("other")])

    async def test_duplicate_parent_provenance_only_looked_up_once(self):
        p = parent()
        self.expansion = ContextExpansion((p, p), ())
        packet = await self.build()
        self.assertEqual(packet, self.original())
        self.lookup.assert_awaited_once()
        self.assertEqual(len(packet["relationships"]), 1)

    async def test_missing_and_ambiguous_notices(self):
        self.expansion = ContextExpansion((), (
            UnresolvedParent("child", "act", "Part", 1,
                             "no same-source passage has the parent section label"),
            UnresolvedParent("second", "act", "Other", 1,
                             "ambiguous parent: 2 same-source passages matched"),
        ))
        packet = await self.build()
        self.assertEqual(packet, self.original())
        self.assertEqual(len(packet["context_resolution_notices"]), 2)

    async def test_parameters_and_untrimmed_settings(self):
        kwargs = dict(top_k=50, source_id=" act ", legal_domain=" Civil ",
                      language=" English ", max_parent_depth=3, max_parent_passages=10)
        packet = await self.build(**kwargs)
        self.assertEqual(packet, self.original(**kwargs))
        self.assertEqual(packet["retrieval_settings"], kwargs)
        self.retrieve.assert_awaited_once_with(
            " query ", top_k=50, source_id=" act ", legal_domain=" Civil ", language=" English ")
        self.expand.assert_awaited_once_with(self.ranked, max_depth=3, max_passages=10)

    async def test_empty_still_expands_and_preserves_packet(self):
        self.ranked.clear()
        packet = await self.build()
        self.assertEqual(packet, self.original())
        self.assertEqual(packet["retrieval_state"], "no_evidence_retrieved")
        self.expand.assert_awaited_once()
        self.lookup.assert_not_awaited()

    async def test_query_validation_precedes_dependencies(self):
        for query in ("", " ", None, 2):
            with self.assertRaisesRegex(ValueError, "query must be a nonblank string"):
                await self.build(query)
        self.retrieve.assert_not_awaited()
        self.expand.assert_not_awaited()

    async def test_retrieval_failure(self):
        error = RuntimeError("retrieval")
        self.retrieve.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            await self.build()
        self.assertIs(caught.exception, error)
        self.expand.assert_not_awaited()

    async def test_expansion_failure(self):
        error = ValueError("expansion")
        self.expand.side_effect = error
        with self.assertRaises(ValueError) as caught:
            await self.build()
        self.assertIs(caught.exception, error)
        self.lookup.assert_not_awaited()

    async def test_provenance_failure(self):
        self.expansion = ContextExpansion((parent(),), ())
        error = LookupError("missing original")
        self.lookup.side_effect = error
        with self.assertRaises(LookupError) as caught:
            await self.build()
        self.assertIs(caught.exception, error)

    async def test_invalid_provenance_and_cross_source_are_not_hidden(self):
        self.expansion = ContextExpansion((parent(),), ())
        self.lookup.side_effect = None
        self.lookup.return_value = ""
        with self.assertRaisesRegex(ValueError, "evidence chunk ID must be nonblank"):
            await self.build()
        self.lookup.return_value = "original-parent"
        self.expansion.parents[0].metadata["source_id"] = "other"
        with self.assertRaisesRegex(ValueError, "crosses legal sources"):
            await self.build()

    async def test_wiring_actual_async_adapters_without_database(self):
        import numpy as np
        embedder = AsyncMock(return_value=np.eye(1, 768)[0])
        search = AsyncMock(return_value=self.ranked)
        lookup = AsyncMock(return_value=[dict(
            chunk_id="parent", text="parent text", metadata=metadata("Part"),
            parent_context=None, paragraph=None)])
        packet = await build_evidence_packet(
            "query", retrieve=partial(retrieve, embedder=embedder, search=search),
            expand_parent_context=partial(expand_parent_context, parent_lookup=lookup),
            original_chunk_lookup=self.lookup,
        )
        self.assertEqual(packet["context_evidence_ids"], ["chunk:parent"])
        lookup.assert_awaited_once_with(source_id="act", section_label="Part")


if __name__ == "__main__":
    unittest.main()
