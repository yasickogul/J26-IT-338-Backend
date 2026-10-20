"""Offline models, frozen migration DDL, and artifact importer tests."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, AsyncMock
import numpy as np
from sqlalchemy.schema import CreateTable, CreateIndex
from sqlalchemy.dialects.postgresql import dialect
from app.shared.corpus.models import Base, LegalDocument, LegalChunk
from app.shared.corpus.ingest import validate_artifacts, import_artifacts


class SchemaTests(unittest.TestCase):
    def test_uuid_and_textual_identity_and_vector_type(self):
        table = LegalChunk.__table__
        self.assertEqual(table.schema, "shared")
        self.assertEqual(str(table.c.id.type), "UUID")
        self.assertEqual(str(table.c.chunk_id.type), "TEXT")
        self.assertTrue(table.c.chunk_id.unique)
        self.assertFalse(table.c.original_chunk_id.nullable)
        self.assertEqual(str(table.c.embedding.type), "VECTOR(768)")
        self.assertTrue(table.c.parent_context.nullable)
        self.assertTrue(table.c.paragraph.nullable)
        self.assertEqual(next(iter(table.c.document_id.foreign_keys)).target_fullname,
                         "shared.legal_documents.id")

    def test_frozen_migration_matches_orm_and_safe_downgrade(self):
        path = Path("app/shared/migrations/versions/shared_0001_corpus.py")
        tree = ast.parse(path.read_text())
        statements = next(ast.literal_eval(n.value) for n in tree.body
                          if isinstance(n, ast.Assign) and n.targets[0].id == "STATEMENTS")
        expected = []
        for table in Base.metadata.sorted_tables:
            expected.append(str(CreateTable(table).compile(dialect=dialect())))
            expected.extend(str(CreateIndex(i).compile(dialect=dialect()))
                            for i in sorted(table.indexes, key=lambda i: i.name))
        self.assertEqual(statements, expected)
        # Execute actual revision functions with only an inert Alembic op.
        namespace = {}
        imports_removed = [n for n in tree.body if not isinstance(n, ast.ImportFrom)]
        op = MagicMock()
        namespace["op"] = op
        exec(compile(ast.Module(imports_removed, []), str(path), "exec"), namespace)
        namespace["upgrade"]()
        calls = [c.args[0] for c in op.execute.call_args_list]
        self.assertIn("pg_extension", calls[0])
        self.assertIn("Existing corpus objects", calls[0])
        self.assertEqual(calls[2:], statements)
        op.reset_mock()
        namespace["downgrade"]()
        calls = [c.args[0] for c in op.execute.call_args_list]
        self.assertIn("ACCESS EXCLUSIVE", calls[0])
        self.assertIn("Refusing to drop nonempty", calls[1])
        self.assertEqual(calls[-2:], ["DROP TABLE shared.legal_chunks", "DROP TABLE shared.legal_documents"])
        self.assertFalse(any("DROP SCHEMA" in c or "DROP EXTENSION" in c for c in calls))


class ImportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.paths = {name + "_path": root / name for name in ("manifest", "input", "corpus", "vector")}
        self.chunk = dict(chunk_id="c1", text="Complete original தமிழ் passage",
                          metadata=dict(source_id="s1", title="Act", section_label="Part > Section",
                                        language="English", legal_domain="Civil",
                                        document_type="act", source_file="act.pdf", act_number="1"))
        self.corpus = [self.chunk]
        self.inputs = [dict(self.chunk, embedding_text=self.chunk["text"])]
        self.vectors = np.eye(1, 768, dtype=np.float32)
        self.save()

    def save(self):
        self.paths["corpus_path"].write_text(json.dumps(self.corpus))
        self.paths["input_path"].write_text(json.dumps(dict(
            inputs=self.inputs, model_name="sentence-transformers/LaBSE",
            dimension=768, derived_input_count=len(self.inputs))))
        self.manifest = dict(model_name="sentence-transformers/LaBSE", dimension=768,
                             count=len(self.inputs),
                             input_file_sha256=hashlib.sha256(self.paths["input_path"].read_bytes()).hexdigest(),
                             original_corpus_sha256=hashlib.sha256(self.paths["corpus_path"].read_bytes()).hexdigest())
        self.paths["manifest_path"].write_text(json.dumps(self.manifest))
        with self.paths["vector_path"].open("wb") as stream:
            np.savez(stream, embeddings=self.vectors,
                     chunk_ids=np.array([x["chunk_id"] for x in self.inputs]))

    def session(self, count=0):
        session = MagicMock()
        session.in_transaction.return_value = False
        result = MagicMock()
        result.scalar_one.return_value = count
        session.execute = AsyncMock(return_value=result)
        session.begin.return_value.__aexit__.return_value = False
        return session

    async def test_validation_preserves_content_and_summary(self):
        docs, rows, summary = validate_artifacts(**self.paths)
        self.assertEqual(summary["chunks"], 1)
        self.assertTrue(summary["dry_run"])
        self.assertEqual(docs[0]["source_id"], "s1")
        self.assertEqual(rows[0]["legal_text"], self.chunk["text"])
        self.assertEqual(rows[0]["original_chunk_id"], "c1")
        self.assertEqual(json.loads(rows[0]["metadata"]), self.chunk["metadata"])

    async def test_hash_and_model_rejection(self):
        for key, value in (("model_name", "other"), ("dimension", 384),
                           ("input_file_sha256", "bad"), ("original_corpus_sha256", "bad")):
            self.paths["manifest_path"].write_text(json.dumps(dict(self.manifest, **{key: value})))
            with self.assertRaises(ValueError):
                validate_artifacts(**self.paths)

    async def test_wrong_vectors(self):
        for vector in (np.zeros((1, 768), dtype=np.float32),
                       np.full((1, 768), np.nan, dtype=np.float32),
                       np.eye(1, 767, dtype=np.float32),
                       np.eye(1, 768, dtype=np.float64)):
            self.vectors = vector
            self.save()
            with self.assertRaises(ValueError):
                validate_artifacts(**self.paths)

    async def test_duplicate_and_changed_text_rejected(self):
        self.inputs.append(self.inputs[0])
        self.save()
        with self.assertRaises(ValueError):
            validate_artifacts(**self.paths)
        self.inputs = [dict(self.inputs[0], text="changed")]
        self.save()
        with self.assertRaises(ValueError):
            validate_artifacts(**self.paths)

    async def test_missing_metadata_and_original_provenance_rejected(self):
        self.inputs = [dict(self.chunk, original_chunk_id="invented", embedding_text=self.chunk["text"])]
        self.save()
        with self.assertRaises(ValueError):
            validate_artifacts(**self.paths)
        self.inputs = [dict(self.chunk, embedding_text=self.chunk["text"])]
        del self.chunk["metadata"]["source_id"]
        self.save()
        with self.assertRaises(ValueError):
            validate_artifacts(**self.paths)

    async def test_transaction_intent_and_no_upsert(self):
        session = self.session()
        summary = await import_artifacts(session=session, **self.paths)
        self.assertFalse(summary["dry_run"])
        session.begin.assert_called_once()
        session.begin.return_value.__aexit__.assert_awaited_once()
        sql = " ".join(str(c.args[0]) for c in session.execute.await_args_list)
        self.assertIn("LOCK TABLE", sql)
        self.assertIn("CAST(:embedding AS vector)", sql)
        self.assertNotIn("ON CONFLICT", sql)
        self.assertNotIn("UPDATE", sql)
        session.close.assert_not_called()

    async def test_existing_data_and_active_transaction_refused(self):
        session = self.session(1)
        with self.assertRaises(ValueError):
            await import_artifacts(session=session, **self.paths)
        self.assertEqual(session.execute.await_count, 2)
        session = self.session()
        session.in_transaction.return_value = True
        with self.assertRaises(ValueError):
            await import_artifacts(session=session, **self.paths)
        session.execute.assert_not_awaited()

    async def test_insert_failure_exits_transaction_for_rollback(self):
        session = self.session()
        result = MagicMock()
        result.scalar_one.return_value = 0
        error = RuntimeError("insert failed")
        session.execute.side_effect = [result, result, result, error]
        with self.assertRaises(RuntimeError):
            await import_artifacts(session=session, **self.paths)
        self.assertIs(session.begin.return_value.__aexit__.call_args.args[0], RuntimeError)
