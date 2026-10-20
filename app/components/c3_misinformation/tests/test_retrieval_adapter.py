import unittest
from unittest.mock import AsyncMock

import numpy as np

from app.components.c3_misinformation.pipeline.retrieval_adapter import retrieve
from app.components.c3_misinformation.pipeline.retrieval_contract import RetrievalResult


def record(chunk_id, distance, similarity):
    return RetrievalResult(
        chunk_id=chunk_id, original_chunk_id=f"original-{chunk_id}",
        text=f"text-{chunk_id}", metadata={"source_id": "source-1"},
        parent_context=None, paragraph=None, distance=distance, similarity=similarity,
    )


class RetrievalAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.vector = np.zeros(768, dtype=np.float32)
        self.vector[0] = 1.0
        self.embedder = AsyncMock(return_value=self.vector)
        self.results = [record("chunk-a", 0.1, 0.9), record("chunk-b", 0.2, 0.8)]
        self.search = AsyncMock(return_value=self.results)

    async def test_valid_query_forwards_vector_filters_and_preserves_results(self):
        found = await retrieve(
            "  legal query  ", embedder=self.embedder, search=self.search,
            top_k=7, source_id="  source-1 ", legal_domain="  civil ", language="  en ",
        )
        self.assertIs(found[0], self.results[0])
        self.assertEqual(found[0].distance, 0.1)
        self.assertEqual(found[0].similarity, 0.9)
        self.embedder.assert_awaited_once_with("  legal query  ")
        vector = self.search.await_args.args[0]
        np.testing.assert_array_equal(vector, self.vector)
        self.assertEqual(self.search.await_args.kwargs, {
            "top_k": 7, "source_id": "source-1", "legal_domain": "civil", "language": "en",
        })

    async def test_default_and_maximum_top_k(self):
        await retrieve("query", embedder=self.embedder, search=self.search)
        self.assertEqual(self.search.await_args.kwargs["top_k"], 5)
        await retrieve("query", embedder=self.embedder, search=self.search, top_k=50)
        self.assertEqual(self.search.await_args.kwargs["top_k"], 50)

    async def test_invalid_top_k_matches_original_contract(self):
        for value in (True, False, 0, 51, 1.0, "5"):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    await retrieve("query", embedder=self.embedder, search=self.search, top_k=value)
        self.embedder.assert_not_awaited()

    async def test_invalid_filters_and_query_are_rejected_before_dependencies(self):
        for kwargs in ({"source_id": "  "}, {"legal_domain": 3}, {"language": ""}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    await retrieve("query", embedder=self.embedder, search=self.search, **kwargs)
        for query in ("", " \t", 42):
            with self.subTest(query=query):
                with self.assertRaises((TypeError, ValueError)):
                    await retrieve(query, embedder=self.embedder, search=self.search)
        self.embedder.assert_not_awaited()

    async def test_embedding_shape_and_nonfinite_validation(self):
        for vector in (np.zeros(767), np.zeros((1, 768)), np.array([np.nan] + [0.0] * 767)):
            with self.subTest(shape=np.asarray(vector).shape):
                self.embedder.reset_mock(return_value=True)
                self.embedder.return_value = vector
                with self.assertRaisesRegex(RuntimeError, "finite 768-dimensional"):
                    await retrieve("query", embedder=self.embedder, search=self.search)
                self.search.assert_not_awaited()

    async def test_embedding_and_search_errors_propagate(self):
        self.embedder.side_effect = RuntimeError("embedding failed")
        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            await retrieve("query", embedder=self.embedder, search=self.search)
        self.embedder.side_effect = None
        self.embedder.return_value = self.vector
        self.search.side_effect = RuntimeError("search failed")
        with self.assertRaisesRegex(RuntimeError, "search failed"):
            await retrieve("query", embedder=self.embedder, search=self.search)

    async def test_empty_search_results_are_returned(self):
        self.search.return_value = []
        self.assertEqual(await retrieve("query", embedder=self.embedder, search=self.search), [])

    async def test_search_order_and_metadata_are_not_rewritten(self):
        self.results = [record("z", 0.9, 0.1), record("a", 0.2, 0.8)]
        self.search.return_value = self.results
        found = await retrieve("query", embedder=self.embedder, search=self.search)
        self.assertEqual([item.chunk_id for item in found], ["z", "a"])
        self.assertEqual(found[0].metadata, {"source_id": "source-1"})

    async def test_search_must_return_retrieval_records(self):
        self.search.return_value = [{"chunk_id": "not-a-contract"}]
        with self.assertRaises(TypeError):
            await retrieve("query", embedder=self.embedder, search=self.search)

    async def test_sync_dependencies_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "awaitable"):
            await retrieve("query", embedder=lambda query: self.vector, search=self.search)
        with self.assertRaisesRegex(TypeError, "awaitable"):
            await retrieve("query", embedder=self.embedder, search=lambda *args, **kwargs: [])


if __name__ == "__main__":
    unittest.main()
