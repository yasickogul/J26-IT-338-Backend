"""Offline tests for lazy C3 worker dependency composition."""

import sys
import threading
import unittest
from types import ModuleType
from unittest.mock import AsyncMock, patch

from app.components.c3_misinformation import worker_dependencies as dependencies
from app.components.c3_misinformation.agents.gemini_adapter import (
    GeminiConfigurationError,
)
from app.components.c3_misinformation.pipeline.labse_query_embedding import MODEL_NAME
from app.components.c3_misinformation.prompts.rag_prompt import ClaimInput
from app.components.c3_misinformation.rag_service_async import ThreadedGeminiProvider


class FakeSession:
    def __init__(self):
        self.execute = AsyncMock()
        self.rollback = AsyncMock()
        self.close = AsyncMock()


class WorkerDependencyTests(unittest.IsolatedAsyncioTestCase):
    def test_composition_is_lazy_and_fixes_trusted_labse_model_identity(self):
        session_factory = unittest.mock.Mock()
        embedder = AsyncMock()
        adapter_factory = unittest.mock.Mock()
        with patch.object(dependencies, "embed_text") as embed_text, \
             patch.object(dependencies, "_get_cached_labse_encoder") as load_labse, \
             patch.object(dependencies, "GeminiAdapter") as adapter_type:
            composed = dependencies.create_worker_dependencies(
                session_factory=session_factory,
                query_embedder=embedder,
                gemini_adapter_factory=adapter_factory,
            )

        embed_text.assert_not_called()
        load_labse.assert_not_called()
        adapter_type.assert_not_called()
        session_factory.assert_not_called()
        self.assertIs(composed.session_factory, session_factory)
        self.assertIs(composed.query_embedder, embedder)
        self.assertIsInstance(composed.gemini_provider, ThreadedGeminiProvider)
        self.assertEqual(composed.config.top_k, 5)
        self.assertEqual(composed.rag_service.model, "gemini-3.5-flash-lite")

    async def test_gemini_provider_factory_is_lazy_and_executes_off_loop(self):
        factory_thread_ids = []
        event_loop_thread = threading.get_ident()

        class FakeAdapter:
            def generate_json(self, prompt, schema):
                return prompt, schema

        def adapter_factory():
            factory_thread_ids.append(threading.get_ident())
            return FakeAdapter()

        provider = ThreadedGeminiProvider(adapter_factory)
        self.assertEqual(factory_thread_ids, [])
        prompt = {"input": "synthetic"}
        schema = {"type": "object"}
        result = await provider.generate_json(prompt, schema)
        self.assertEqual(result, (prompt, schema))
        self.assertNotEqual(factory_thread_ids[0], event_loop_thread)

    async def test_unavailable_gemini_configuration_propagates_without_fallback(self):
        class UnavailableAdapter:
            def generate_json(self, _prompt, _schema):
                raise GeminiConfigurationError("google-genai is not installed")

        provider = ThreadedGeminiProvider(UnavailableAdapter)
        with self.assertRaisesRegex(
            GeminiConfigurationError, "google-genai is not installed",
        ):
            await provider.generate_json({"prompt": "fixture"}, {"type": "object"})

    async def test_packet_builder_uses_c3_corpus_read_only_and_closes_session(self):
        session = FakeSession()
        factory = unittest.mock.Mock(return_value=session)
        embedder = AsyncMock()
        config = dependencies.WorkerRuntimeConfig(
            top_k=7, source_id="court-a", legal_domain="civil", language="en",
        )
        composed = dependencies.create_worker_dependencies(
            session_factory=factory, query_embedder=embedder, config=config,
        )
        expected_packet = {"evidence": [{"chunk_id": "fixture"}]}
        with patch.object(
            dependencies, "build_c3_evidence_packet",
            new=AsyncMock(return_value=expected_packet),
        ) as build:
            result = await composed.evidence_packet_builder(
                "legal query",
                top_k=config.top_k,
                source_id=config.source_id,
                legal_domain=config.legal_domain,
                language=config.language,
            )

        self.assertIs(result, expected_packet)
        session.execute.assert_awaited_once()
        self.assertEqual(str(session.execute.await_args.args[0]), "SET TRANSACTION READ ONLY")
        session.rollback.assert_awaited_once()
        session.close.assert_awaited_once()
        build.assert_awaited_once_with(
            "legal query",
            session=session,
            embedder=embedder,
            query_model=MODEL_NAME,
            top_k=7,
            source_id="court-a",
            legal_domain="civil",
            language="en",
        )

    async def test_packet_builder_closes_session_and_propagates_retrieval_failure(self):
        session = FakeSession()
        composed = dependencies.create_worker_dependencies(
            session_factory=lambda: session,
            query_embedder=AsyncMock(),
        )
        failure = RuntimeError("private retrieval failure")
        with patch.object(
            dependencies, "build_c3_evidence_packet",
            new=AsyncMock(side_effect=failure),
        ), self.assertRaisesRegex(RuntimeError, "private retrieval failure") as caught:
            await composed.evidence_packet_builder(
                "legal query", top_k=5, source_id=None,
                legal_domain=None, language=None,
            )
        self.assertIs(caught.exception, failure)
        session.rollback.assert_awaited_once()
        session.close.assert_awaited_once()

    async def test_analyzer_binds_claim_input_top_k_and_filters(self):
        composed = dependencies.create_worker_dependencies(
            session_factory=lambda: FakeSession(),
            query_embedder=AsyncMock(),
            config=dependencies.WorkerRuntimeConfig(
                source_id="court-a", legal_domain="civil", language="en",
            ),
        )
        expected = object()
        with patch.object(composed.rag_service, "run", new=AsyncMock(return_value=expected)) as run:
            result = await composed.analyzer("a legal claim", top_k=9)
        self.assertIs(result, expected)
        run.assert_awaited_once()
        input_data = run.await_args.args[0]
        self.assertIsInstance(input_data, ClaimInput)
        self.assertEqual(input_data.claim, "a legal claim")
        self.assertEqual(run.await_args.kwargs, {
            "top_k": 9,
            "source_id": "court-a",
            "legal_domain": "civil",
            "language": "en",
        })

    def test_configuration_rejects_invalid_runtime_filters_and_top_k(self):
        for kwargs in (
            {"top_k": True}, {"top_k": 0}, {"top_k": 51},
            {"source_id": " "}, {"legal_domain": 2}, {"language": ""},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                dependencies.WorkerRuntimeConfig(**kwargs)

        with self.assertRaisesRegex(TypeError, "session_factory must be callable"):
            dependencies.create_worker_dependencies(session_factory=None)

    def test_configured_composer_fails_explicitly_without_existing_database_factory(self):
        from app.core import database

        with patch.object(database, "SessionLocal", None), self.assertRaisesRegex(
            dependencies.WorkerDependencyConfigurationError,
            "existing DATABASE_URL-backed session factory",
        ):
            dependencies.create_configured_worker_dependencies()

    def test_configured_composer_reuses_sessionmaker_without_creating_a_session(self):
        from app.core import database

        session_factory = unittest.mock.Mock()
        with patch.object(database, "SessionLocal", session_factory):
            composed = dependencies.create_configured_worker_dependencies(
                query_embedder=AsyncMock(),
            )
        self.assertIs(composed.session_factory, session_factory)
        session_factory.assert_not_called()

    async def test_default_query_embedder_moves_sync_labse_work_to_thread(self):
        composed = dependencies.create_worker_dependencies(
            session_factory=lambda: FakeSession(),
        )
        event_loop_thread = threading.get_ident()
        work_thread_ids = []

        def encode(query, *, encoder):
            work_thread_ids.append(threading.get_ident())
            self.assertIsNotNone(encoder)
            return query

        async def embed_from_packet(query, **kwargs):
            return await kwargs["embedder"](query)

        with patch.object(dependencies, "_get_cached_labse_encoder", return_value=object()), \
             patch.object(dependencies, "embed_text", side_effect=encode):
            builder = composed.evidence_packet_builder
            with patch.object(
                dependencies, "build_c3_evidence_packet",
                new=AsyncMock(side_effect=embed_from_packet),
            ):
                encoded = await builder(
                    "query", top_k=5, source_id=None,
                    legal_domain=None, language=None,
                )
        self.assertEqual(encoded, "query")
        self.assertTrue(work_thread_ids)
        self.assertNotEqual(work_thread_ids[0], event_loop_thread)

    def test_labse_loader_requires_a_cached_snapshot_and_valid_dimension(self):
        class Encoder:
            def get_embedding_dimension(self):
                return 768

        hub = ModuleType("huggingface_hub")
        hub.snapshot_download = unittest.mock.Mock(return_value="/cached/labse")
        transformers = ModuleType("sentence_transformers")
        transformers.SentenceTransformer = unittest.mock.Mock(return_value=Encoder())
        with patch.dict(
            sys.modules,
            {"huggingface_hub": hub, "sentence_transformers": transformers},
        ), patch.object(dependencies, "_labse_encoder", None):
            encoder = dependencies._get_cached_labse_encoder()

        self.assertIsInstance(encoder, Encoder)
        hub.snapshot_download.assert_called_once_with(
            MODEL_NAME,
            allow_patterns=dependencies.LABSE_SNAPSHOT_ALLOW_PATTERNS,
            local_files_only=True,
        )
        transformers.SentenceTransformer.assert_called_once_with(
            "/cached/labse", device="cpu",
        )


if __name__ == "__main__":
    unittest.main()
