"""SQL intent and source equivalence only; no PostgreSQL connections."""
import ast
from pathlib import Path
from dataclasses import asdict
import unittest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import dialect
from app.components.c3_misinformation.pipeline import postgres_vector_search as pg
from app.components.c3_misinformation.pipeline import retrieval_contract as contract


def row():
    return dict(chunk_id="c", original_chunk_id="original", text="Complete தமிழ் text",
                metadata={"source_id": "s"}, parent_context=None, paragraph=None,
                distance=0.2, similarity=0.8)


class VectorSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = MagicMock()
        self.session.execute = AsyncMock(return_value=MagicMock())
        self.session.execute.return_value.mappings.return_value = [row()]
        self.vector = np.eye(1, 768)[0]

    async def search(self, **kwargs):
        return await pg.search(self.vector, session=self.session,
                               **dict(corpus_contract=pg.COMPNET_CORPUS_CONTRACT,
                                      query_model=pg.MODEL_NAME, **kwargs))

    async def test_exact_source_sql_parameters_and_records(self):
        source_path = Path("/home/sano_01/fyp/backend/app/services/retrieval.py")
        if not source_path.exists():
            self.skipTest("CompNet source unavailable")
        function = next(n for n in ast.parse(source_path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "retrieve")
        namespace = dict(vars(contract), np=np, text=text, Session=object,
                         MODEL_NAME=pg.MODEL_NAME, embed_text=lambda q: self.vector)
        exec(compile(ast.Module([function], []), str(source_path), "exec"), namespace)
        for top_k in (5, 50):
            options = dict(top_k=top_k, source_id=" s ' ", legal_domain=" Civil ", language=" English ")
            db = MagicMock()
            db.execute.return_value.mappings.return_value = [row()]
            expected = namespace["retrieve"]("query", db, **options)
            actual = await self.search(**options)
            self.assertEqual([asdict(r) for r in actual], [asdict(r) for r in expected])
            actual_sql, actual_params = self.session.execute.call_args.args
            original_sql, original_params = db.execute.call_args.args
            self.assertEqual(" ".join(str(actual_sql).split()), " ".join(str(original_sql).split()))
            self.assertEqual(actual_params, original_params)
            compiled = str(actual_sql.compile(dialect=dialect()))
            self.assertIn("<=> CAST(", compiled)
            self.assertIn("1 - distance AS similarity", compiled)
            self.assertIn("ORDER BY distance ASC, chunk_id ASC", compiled)
            self.assertNotIn(" s ' ", compiled)

    async def test_empty_defaults_and_no_transaction_management(self):
        self.session.execute.return_value.mappings.return_value = []
        self.assertEqual(await self.search(), [])
        sql, params = self.session.execute.call_args.args
        self.assertEqual(params["top_k"], 5)
        self.assertEqual(params["embedding_model"], "sentence-transformers/LaBSE")
        self.assertNotIn("source_id", params)
        for name in ("begin", "commit", "rollback", "close"):
            getattr(self.session, name).assert_not_called()

    async def test_invalid_vectors_before_sql(self):
        for vector in (np.zeros(767), np.zeros((1, 768)), [float("nan")]*768,
                       [float("inf")]*768, ["x"]*768, [1j]*768):
            self.vector = vector
            with self.assertRaises(RuntimeError):
                await self.search()
        self.session.execute.assert_not_awaited()

    async def test_invalid_limits_filters(self):
        for options in ({"top_k": True}, {"top_k": 0}, {"top_k": 51},
                        {"top_k": 1.0}, {"language": ""}, {"source_id": 1},
                        {"legal_domain": " "}):
            with self.assertRaises((ValueError, TypeError)):
                await self.search(**options)
        self.session.execute.assert_not_awaited()

    async def test_contract_and_model_are_explicit(self):
        for mapping, model in ((None, pg.MODEL_NAME), ("shared.legal_chunks", pg.MODEL_NAME),
                               (pg.COMPNET_CORPUS_CONTRACT, "other-768-model")):
            with self.assertRaises(ValueError):
                await pg.search(self.vector, session=self.session,
                                corpus_contract=mapping, query_model=model)
        with self.assertRaises(TypeError):
            await pg.search(self.vector, session=self.session)
        self.session.execute.assert_not_awaited()

    async def test_malformed_metadata_or_missing_columns_fail(self):
        for change in (dict(metadata=None), dict(metadata=[]), dict(original_chunk_id=""),
                       dict(parent_context=[])):
            self.session.execute.return_value.mappings.return_value = [dict(row(), **change)]
            with self.assertRaises(ValueError):
                await self.search()
        bad = row()
        del bad["original_chunk_id"]
        self.session.execute.return_value.mappings.return_value = [bad]
        with self.assertRaises(KeyError):
            await self.search()

    async def test_storage_errors_propagate(self):
        error = RuntimeError("storage unavailable")
        self.session.execute.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            await self.search()
        self.assertIs(caught.exception, error)
