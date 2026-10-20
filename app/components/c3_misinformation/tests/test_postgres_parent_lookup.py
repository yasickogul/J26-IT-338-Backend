import ast
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.dialects.postgresql import dialect
from functools import partial
from app.components.c3_misinformation.pipeline.postgres_parent_lookup import lookup_parents
from app.components.c3_misinformation.pipeline.postgres_vector_search import COMPNET_CORPUS_CONTRACT
from app.components.c3_misinformation.pipeline.context_expansion_adapter import expand_parent_context
from app.components.c3_misinformation.pipeline.retrieval_contract import RetrievalResult


class ParentLookupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = MagicMock()
        self.session.execute = AsyncMock(return_value=MagicMock())
        self.rows = [dict(chunk_id=cid, text="Full text", metadata={
            "source_id": " s ", "section_label": "Part"}, parent_context=None, paragraph=None)
            for cid in ("a", "z")]
        self.session.execute.return_value.mappings.return_value.all.return_value = self.rows

    async def lookup(self):
        return await lookup_parents(source_id=" s ", section_label="Part",
                                    session=self.session, corpus_contract=COMPNET_CORPUS_CONTRACT)

    async def test_all_candidates_exact_source_sql_and_order(self):
        self.assertEqual(await self.lookup(), self.rows)
        sql, params = self.session.execute.call_args.args
        self.assertEqual(params, {"source_id": " s ", "section_label": "Part"})
        compiled = str(sql.compile(dialect=dialect()))
        self.assertIn("ORDER BY chunk_id ASC", compiled)
        self.assertNotIn("LIMIT", compiled)
        path = Path("/home/sano_01/fyp/backend/app/services/context_expansion.py")
        if not path.exists():
            self.skipTest("CompNet source unavailable")
        original = next(n.value for n in ast.walk(ast.parse(path.read_text()))
                        if isinstance(n, ast.Constant) and isinstance(n.value, str)
                        and "SELECT chunk_id" in n.value)
        self.assertEqual(" ".join(str(sql).split()), " ".join(original.split()))

    async def test_ambiguity_preserved_in_actual_expansion(self):
        child = RetrievalResult("child", "o", "text", {
            "source_id": " s ", "section_label": "Part > Child"}, None, None, .1, .9)
        result = await expand_parent_context([child], parent_lookup=partial(
            lookup_parents, session=self.session, corpus_contract=COMPNET_CORPUS_CONTRACT))
        self.assertEqual(result.parents, ())
        self.assertEqual(result.unresolved[0].reason, "ambiguous parent: 2 same-source passages matched")

    async def test_empty_and_no_session_management(self):
        self.session.execute.return_value.mappings.return_value.all.return_value = []
        self.assertEqual(await self.lookup(), [])
        for name in ("begin", "commit", "rollback", "close"):
            getattr(self.session, name).assert_not_called()

    async def test_bad_metadata_and_cross_source_fail(self):
        for metadata in (None, {}, {"source_id": "other", "section_label": "Part"},
                         {"source_id": " s ", "section_label": "Other"}):
            self.session.execute.return_value.mappings.return_value.all.return_value = [
                dict(self.rows[0], metadata=metadata)]
            with self.assertRaises(ValueError):
                await self.lookup()

    async def test_unknown_mapping_and_storage_error(self):
        with self.assertRaises(ValueError):
            await lookup_parents(source_id="s", section_label="Part",
                                 session=self.session, corpus_contract="shared")
        self.session.execute.assert_not_awaited()
        error = RuntimeError("read failure")
        self.session.execute.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            await self.lookup()
        self.assertIs(caught.exception, error)
