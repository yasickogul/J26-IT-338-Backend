"""Offline checks for multilingual claim adaptation at the RAG boundary."""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.components.c3_misinformation.pipeline.claim_translation import (
    TranslationProviderError,
    TranslationProviderResponse,
)
from app.components.c3_misinformation.pipeline.multilingual_claim_adapter import (
    MultilingualClaimAdapter,
)
from app.components.c3_misinformation.prompts.rag_prompt import ClaimInput


class MockTranslationProvider:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def translate(self, text, *, source_language, target_language):
        self.calls.append((text, source_language, target_language))
        if self.error is not None:
            raise self.error
        return self.response


class MultilingualClaimAdapterTests(unittest.TestCase):
    def test_enabled_default_reuses_shared_lazy_identifier(self):
        identifier = SimpleNamespace()
        with patch(
            "app.components.c3_misinformation.pipeline.multilingual_claim_adapter."
            "get_fasttext_language_identifier",
            return_value=identifier,
        ) as factory:
            adapter = MultilingualClaimAdapter(detection_enabled=True)
        self.assertIs(adapter._language_identifier, identifier)
        factory.assert_called_once_with()

    def test_english_claim_keeps_original_retrieval_text(self):
        provider = MockTranslationProvider(
            response=TranslationProviderResponse("Should not be used", "mock:english"),
        )
        adapter = MultilingualClaimAdapter(
            translation_provider=provider,
            detection_enabled=True,
            language_identifier=SimpleNamespace(
                identify=lambda text: SimpleNamespace(label="en", model_name="fake-model", score=0.9),
            ),
        )

        request = adapter.prepare_input(ClaimInput("The law applies."))

        self.assertEqual(request, ClaimInput(claim="The law applies.", retrieval_query="The law applies."))
        self.assertEqual(provider.calls, [])

    def test_sinhala_claim_uses_translated_retrieval_query_without_replacing_claim(self):
        original = "මෙම නීතිය අදාළ වේ."
        provider = MockTranslationProvider(
            response=TranslationProviderResponse("This law applies.", "mock:si-en:test-v1"),
        )
        adapter = MultilingualClaimAdapter(
            translation_provider=provider,
            detection_enabled=True,
            language_identifier=SimpleNamespace(
                identify=lambda text: SimpleNamespace(label="si", model_name="fake-model", score=0.9),
            ),
        )

        request = adapter.prepare_input(ClaimInput(original))

        self.assertEqual(request.claim, original)
        self.assertEqual(request.retrieval_query, "This law applies.")
        self.assertEqual(provider.calls, [(original, "si", "en")])

    def test_failed_translation_keeps_original_query_and_does_not_fabricate_translation(self):
        original = "தமிழ் உரை." 
        provider = MockTranslationProvider(
            error=TranslationProviderError("provider_error"),
        )
        adapter = MultilingualClaimAdapter(
            translation_provider=provider,
            detection_enabled=True,
            language_identifier=SimpleNamespace(
                identify=lambda text: SimpleNamespace(label="ta", model_name="fake-model", score=0.9),
            ),
        )

        request = adapter.prepare_input(ClaimInput(original))

        self.assertEqual(request.claim, original)
        self.assertEqual(request.retrieval_query, original)
        self.assertEqual(provider.calls, [(original, "ta", "en")])

    def test_explicit_retrieval_query_is_preserved(self):
        adapter = MultilingualClaimAdapter(
            translation_provider=MockTranslationProvider(
                response=TranslationProviderResponse("translated", "must-not-run"),
            ),
            detection_enabled=True,
        )

        request = adapter.prepare_input(ClaimInput("Original claim text.", retrieval_query="Explicit normalized query."))

        self.assertEqual(request.retrieval_query, "Explicit normalized query.")
        self.assertEqual(request.claim, "Original claim text.")

    def test_unexpected_provider_failure_is_safe(self):
        original = "தமிழ் உரை."

        class BrokenProvider:
            def translate(self, *args, **kwargs):
                raise RuntimeError("provider network failure")

        adapter = MultilingualClaimAdapter(
            translation_provider=BrokenProvider(),
            detection_enabled=True,
            language_identifier=SimpleNamespace(
                identify=lambda text: SimpleNamespace(label="ta", model_name="fake-model", score=0.9),
            ),
        )
        request = adapter.prepare_input(ClaimInput(original))
        self.assertEqual(request, ClaimInput(claim=original, retrieval_query=original))
        async_request = asyncio.run(adapter.aprepare_input(ClaimInput(original)))
        self.assertEqual(async_request, request)

    def test_translation_cancellation_propagates_through_sync_and_async_adapter(self):
        error = asyncio.CancelledError()
        adapter = MultilingualClaimAdapter(
            translation_provider=MockTranslationProvider(error=error),
            detection_enabled=True,
            language_identifier=SimpleNamespace(
                identify=lambda text: SimpleNamespace(label="ta", model_name="fake-model", score=0.9),
            ),
        )
        original = ClaimInput("தமிழ் உரை.")
        with self.assertRaises(asyncio.CancelledError) as caught:
            adapter.prepare_input(original)
        self.assertIs(caught.exception, error)
        with self.assertRaises(asyncio.CancelledError) as caught:
            asyncio.run(adapter.aprepare_input(original))
        self.assertIs(caught.exception, error)

    async def _async_prepare(self):
        calls = []
        loop_thread = threading.get_ident()

        class Provider:
            def translate(self, text, *, source_language, target_language):
                calls.append(threading.get_ident())
                return TranslationProviderResponse("Translated claim.", "fixture")

        adapter = MultilingualClaimAdapter(
            translation_provider=Provider(),
            detection_enabled=True,
            language_identifier=SimpleNamespace(
                identify=lambda text: SimpleNamespace(label="si", model_name="fake-model", score=0.9),
            ),
        )
        result = await adapter.aprepare_input(ClaimInput("සිංහල ප්‍රකාශය."))
        self.assertEqual(result.retrieval_query, "Translated claim.")
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(calls[0], loop_thread)

    def test_async_preparation_offloads_provider(self):
        import asyncio
        asyncio.run(self._async_prepare())


class MultilingualRagServiceIntegrationTests(unittest.TestCase):
    def test_async_rag_service_applies_multilingual_adapter_before_retrieval(self):
        import asyncio

        async def exercise():
            original = "මෙම නීතිය අදාළ වේ."
            provider = MockTranslationProvider(
                response=TranslationProviderResponse("This law applies.", "mock:si-en:test-v1"),
            )
            adapter = MultilingualClaimAdapter(
                translation_provider=provider,
                detection_enabled=True,
                language_identifier=SimpleNamespace(
                    identify=lambda text: SimpleNamespace(label="si", model_name="fake-model", score=0.9),
                ),
            )

            class FakeProvider:
                def __init__(self):
                    self.calls = []

                async def generate_json(self, prompt, schema):
                    self.calls.append((prompt, schema))
                    return SimpleNamespace(
                        response={
                            "claim_verdict": "Misleading",
                            "citations": ["chunk:CHILD-001"],
                            "insufficient_evidence": False,
                            "explanation": "Fixture explanation.",
                        },
                        validation={"status": "passed", "validator": "Draft202012Validator"},
                        model="gemini-fixture",
                        usage={"prompt_token_count": 10, "total_token_count": 15},
                        attempts=1,
                        finish_reason="STOP",
                        limitations=[],
                    )

            async def evidence_packet_builder(query, **kwargs):
                self.assertEqual(query, "This law applies.")
                return {
                    "query": query,
                    "evidence": [{"evidence_id": "chunk:CHILD-001", "text": "Fixture evidence."}],
                    "relationships": [],
                    "context_resolution_notices": [],
                    "limitations": [],
                    "legal_coverage": {},
                }

            from app.components.c3_misinformation.rag_service_async import (
                AsyncRagService,
            )

            rag = AsyncRagService(
                evidence_packet_builder=evidence_packet_builder,
                provider=FakeProvider(),
                token_validator=lambda values: [1] * len(values),
                multilingual_adapter=adapter,
            )
            result = await rag.run(ClaimInput(original))
            self.assertEqual(result.evidence_packet["query"], "This law applies.")
            self.assertEqual(result.response["claim_verdict"], "Misleading")
            self.assertEqual(provider.calls, [(original, "si", "en")])

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
