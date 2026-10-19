"""Shared mapping and full packet wiring, with mocked SQL results only."""
import unittest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from app.components.c3_misinformation.pipeline.shared_corpus_evidence import build_shared_evidence_packet
from app.components.c3_misinformation.pipeline.postgres_vector_search import search, SHARED_CORPUS_CONTRACT
from app.components.c3_misinformation.pipeline.labse_query_embedding import MODEL_NAME


def result(rows=None, scalar=None):
    value = MagicMock()
    value.mappings.return_value.__iter__.return_value = iter(rows or [])
    value.mappings.return_value.all.return_value = rows or []
    value.scalar_one.return_value = scalar
    return value


class SharedCorpusTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_shared_packet_preserves_citations_and_filters(self):
        child = dict(chunk_id="text-child", original_chunk_id="original-child", text="Full text",
                     metadata={"source_id": "act", "section_label": "Part > Section", "title": "Act"},
                     parent_context=None, paragraph="1", distance=.2, similarity=.8)
        parent = dict(chunk_id="text-parent", text="Full parent", metadata={
            "source_id": "act", "section_label": "Part", "title": "Act"},
            parent_context=None, paragraph=None)
        session = MagicMock()
        session.execute = AsyncMock(side_effect=[result([child]), result([parent]), result(scalar="original-parent")])
        packet = await build_shared_evidence_packet(
            "query", session=session, embedder=AsyncMock(return_value=np.eye(1, 768)[0]),
            query_model=MODEL_NAME, source_id=" act ", legal_domain=" Civil ", language=" English ",
        )
        self.assertEqual(packet["evidence"][-1]["original_chunk_id"], "original-parent")
        self.assertEqual(packet["relationships"][0]["child_evidence_id"], "chunk:text-child")
        self.assertEqual(packet["ranked_evidence"][0]["similarity"], .8)
        for invocation in session.execute.await_args_list:
            self.assertIn("shared.legal_chunks", str(invocation.args[0]))
            self.assertNotIn("public.legal_chunks", str(invocation.args[0]))
        sql, params = session.execute.await_args_list[0].args
        self.assertEqual(params["embedding_model"], MODEL_NAME)
        self.assertEqual(params["source_id"], "act")
        self.assertEqual(params["legal_domain"], "Civil")
        self.assertEqual(params["language"], "English")
        self.assertIn("ORDER BY distance ASC, chunk_id ASC", str(sql))
        self.assertNotIn("ORDER BY distance ASC, id", str(sql))
        for method in ("commit", "rollback", "close", "begin"):
            getattr(session, method).assert_not_called()

    async def test_shared_vector_normalization_and_identity(self):
        session = MagicMock()
        session.execute = AsyncMock()
        for vector, model in ((np.zeros(768), MODEL_NAME), (np.eye(1, 768)[0], "other")):
            with self.assertRaises(ValueError):
                await search(vector, session=session, corpus_contract=SHARED_CORPUS_CONTRACT,
                             query_model=model)
        session.execute.assert_not_awaited()

    async def test_missing_and_ambiguous_parents(self):
        child = dict(chunk_id="c", original_chunk_id="o", text="Full", metadata={
            "source_id": "s", "section_label": "P > C"}, parent_context=None, paragraph=None,
            distance=0., similarity=1.)
        parent = dict(chunk_id="p", text="Parent", metadata={
            "source_id": "s", "section_label": "P"}, parent_context=None, paragraph=None)
        for parents, reason in (([], "no same-source"), ([parent, dict(parent, chunk_id="p2")], "ambiguous")):
            session = MagicMock()
            session.execute = AsyncMock(side_effect=[result([child]), result(parents)])
            packet = await build_shared_evidence_packet(
                "query", session=session, embedder=AsyncMock(return_value=np.eye(1, 768)[0]),
                query_model=MODEL_NAME)
            self.assertIn(reason, packet["context_resolution_notices"][0]["reason"])
            self.assertEqual(session.execute.await_count, 2)
