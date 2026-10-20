"""Offline synthetic checks for the independent C3 corpus."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from app.components.c3_misinformation.corpus_import import (
    import_c3_artifacts,
    validate_c3_artifacts,
)
from app.components.c3_misinformation.pipeline.c3_corpus_evidence import (
    build_c3_evidence_packet,
)
from app.components.c3_misinformation.pipeline.labse_query_embedding import MODEL_NAME


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(rows=None, scalar=None):
    value = MagicMock()
    value.mappings.return_value.__iter__.return_value = iter(rows or [])
    value.mappings.return_value.all.return_value = rows or []
    value.scalar_one.return_value = scalar
    return value


class C3CorpusImporterTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict[str, Path]:
        for directory in ("legal_documents", "chunks", "embeddings", "manifests"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        (root / "legal_documents/fixture.pdf").write_bytes(b"reviewed source fixture")
        metadata = {
            "source_id": "ACT-001",
            "title": "Fixture Act",
            "section_label": "Section 1",
            "language": "English",
            "legal_domain": "Civil",
            "document_type": "act",
            "source_file": "fixture.pdf",
            "act_number": "1 of 2026",
        }
        chunk = {
            "chunk_id": "ACT-001-0001",
            "text": "A reviewed synthetic legal passage.",
            "metadata": metadata,
        }
        corpus = root / "chunks/legal_chunks.json"
        inputs = root / "embeddings/prepared_inputs.json"
        manifest = root / "manifests/manifest.json"
        vectors = root / "embeddings/labse_vectors.npz"
        corpus.write_text(json.dumps({"chunks": [chunk]}), encoding="utf-8")
        prepared = {
            "model_name": MODEL_NAME,
            "dimension": 768,
            "derived_input_count": 1,
            "inputs": [{**chunk, "embedding_text": chunk["text"]}],
        }
        inputs.write_text(json.dumps(prepared), encoding="utf-8")
        manifest.write_text(json.dumps({
            "model_name": MODEL_NAME,
            "dimension": 768,
            "count": 1,
            "input_file_sha256": _sha(inputs),
            "original_corpus_sha256": _sha(corpus),
        }), encoding="utf-8")
        vector = np.zeros((1, 768), dtype=np.float32)
        vector[0, 0] = 1.0
        np.savez(vectors, embeddings=vector, chunk_ids=np.array(["ACT-001-0001"]))
        return {
            "data_root": root,
            "manifest_path": Path("manifests/manifest.json"),
            "input_path": Path("embeddings/prepared_inputs.json"),
            "corpus_path": Path("chunks/legal_chunks.json"),
            "vector_path": Path("embeddings/labse_vectors.npz"),
        }

    def test_validates_copied_source_chunks_and_vectors_without_database(self):
        with TemporaryDirectory() as directory:
            sources, chunks, summary = validate_c3_artifacts(
                **self.fixture(Path(directory)),
            )
        self.assertEqual(summary["sources"], 1)
        self.assertEqual(summary["chunks"], 1)
        self.assertTrue(summary["dry_run"])
        self.assertEqual(sources[0]["local_source_path"], "legal_documents/fixture.pdf")
        self.assertEqual(len(sources[0]["source_file_sha256"]), 64)
        self.assertEqual(chunks[0]["embedding_model"], MODEL_NAME)
        self.assertEqual(chunks[0]["original_chunk_id"], "ACT-001-0001")

    def test_rejects_artifacts_outside_root_and_missing_source_file(self):
        with TemporaryDirectory() as directory, TemporaryDirectory() as outside:
            paths = self.fixture(Path(directory))
            external = Path(outside) / "manifest.json"
            external.write_text("{}", encoding="utf-8")
            paths["manifest_path"] = external
            with self.assertRaisesRegex(ValueError, "inside the configured data directory"):
                validate_c3_artifacts(**paths)

            paths = self.fixture(Path(directory))
            (Path(directory) / "legal_documents/fixture.pdf").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_c3_artifacts(**paths)

    def test_database_import_requires_explicit_authorization_before_validation(self):
        session = MagicMock()
        session.execute = AsyncMock()

        async def exercise():
            with self.assertRaises(PermissionError):
                await import_c3_artifacts(
                    session=session,
                    authorization="",
                    manifest_path="missing",
                    input_path="missing",
                    corpus_path="missing",
                    vector_path="missing",
                )

        import asyncio

        asyncio.run(exercise())
        session.execute.assert_not_awaited()


class C3CorpusEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_packet_queries_only_c3_legal_chunks(self):
        child = {
            "chunk_id": "child",
            "original_chunk_id": "original-child",
            "text": "Complete child text",
            "metadata": {
                "source_id": "act",
                "section_label": "Part > Section",
                "title": "Act",
            },
            "parent_context": None,
            "paragraph": None,
            "distance": 0.2,
            "similarity": 0.8,
        }
        parent = {
            "chunk_id": "parent",
            "text": "Complete parent text",
            "metadata": {"source_id": "act", "section_label": "Part", "title": "Act"},
            "parent_context": None,
            "paragraph": None,
        }
        session = MagicMock()
        session.execute = AsyncMock(side_effect=[
            _result([child]), _result([parent]), _result(scalar="original-parent"),
        ])
        packet = await build_c3_evidence_packet(
            "query",
            session=session,
            embedder=AsyncMock(return_value=np.eye(1, 768)[0]),
            query_model=MODEL_NAME,
        )
        self.assertEqual(packet["evidence"][-1]["original_chunk_id"], "original-parent")
        for invocation in session.execute.await_args_list:
            sql = str(invocation.args[0])
            self.assertIn("c3.legal_chunks", sql)
            self.assertNotIn("shared.legal_chunks", sql)
            self.assertNotIn("public.legal_chunks", sql)


if __name__ == "__main__":
    unittest.main()
