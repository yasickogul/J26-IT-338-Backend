import ast
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.exc import NoResultFound, MultipleResultsFound
from sqlalchemy.dialects.postgresql import dialect
from app.components.c3_misinformation.pipeline.postgres_provenance_lookup import lookup_original_chunk_id
from app.components.c3_misinformation.pipeline.postgres_vector_search import COMPNET_CORPUS_CONTRACT


class ProvenanceLookupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = MagicMock()
        self.session.execute = AsyncMock(return_value=MagicMock())
        self.session.execute.return_value.scalar_one.return_value = "CON-001-0004"

    async def lookup(self):
        return await lookup_original_chunk_id("CON-001-0004-a", session=self.session,
                                              corpus_contract=COMPNET_CORPUS_CONTRACT)

    async def test_exact_id_and_source_statement(self):
        self.assertEqual(await self.lookup(), "CON-001-0004")
        sql, params = self.session.execute.call_args.args
        self.assertEqual(params, {"chunk_id": "CON-001-0004-a"})
        self.assertIn("WHERE chunk_id =", str(sql.compile(dialect=dialect())))
        path = Path("/home/sano_01/fyp/backend/app/services/evidence_packet.py")
        if not path.exists():
            self.skipTest("CompNet source unavailable")
        original = next(n.value for n in ast.walk(ast.parse(path.read_text()))
                        if isinstance(n, ast.Constant) and isinstance(n.value, str)
                        and "SELECT original_chunk_id" in n.value)
        self.assertEqual(str(sql), original)
        for name in ("begin", "commit", "rollback", "close"):
            getattr(self.session, name).assert_not_called()

    async def test_missing_and_ambiguous_propagate_without_fallback(self):
        for error in (NoResultFound(), MultipleResultsFound()):
            self.session.execute.return_value.scalar_one.side_effect = error
            with self.assertRaises(type(error)) as caught:
                await self.lookup()
            self.assertIs(caught.exception, error)

    async def test_null_blank_or_nonstring_provenance_fails(self):
        for value in (None, "", " ", 123):
            self.session.execute.return_value.scalar_one.return_value = value
            with self.assertRaises(ValueError):
                await self.lookup()

    async def test_mapping_gate_and_storage_errors(self):
        with self.assertRaises(ValueError):
            await lookup_original_chunk_id("c", session=self.session, corpus_contract=None)
        self.session.execute.assert_not_awaited()
        error = RuntimeError("storage")
        self.session.execute.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            await self.lookup()
        self.assertIs(caught.exception, error)
