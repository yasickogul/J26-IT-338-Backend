"""Offline tests for async RAG orchestration and source compatibility."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.components.c3_misinformation.agents.gemini_adapter import (
    DEFAULT_MODEL,
    GeminiAdapterError,
    GeminiConfigurationError,
    GeminiInvalidResponseError,
    GeminiRequestPreparationError,
    GeminiRequestTooLargeError,
    GeminiGeneration,
)
from app.components.c3_misinformation.pipeline.claim_input_tokens import (
    EmbeddingInputTooLongError,
    TokenizerUnavailableError,
)
from app.components.c3_misinformation.prompts.rag_prompt import (
    ClaimInput,
    QuestionInput,
)
from app.components.c3_misinformation.rag_contract import (
    RagInputError,
    RagInputTooLongError,
    RagPreparationError,
    RagProviderError,
    RagRequestTooLargeError,
    RagResponseValidationError,
    RagRetrievalError,
)
from app.components.c3_misinformation.rag_response import serialize_rag_result
from app.components.c3_misinformation.rag_service_async import (
    AsyncRagService,
    ThreadedGeminiProvider,
)
from app.components.c3_misinformation.worker_rag_adapter import (
    create_worker_rag_analyzer,
)


SOURCE_BACKEND = Path("/home/sano_01/fyp/backend")
SOURCE_SERVICE = SOURCE_BACKEND / "app/services/rag_service.py"


def packet(query: str, *, empty: bool = False) -> dict:
    passages = [] if empty else [{
        "evidence_id": "chunk:CHILD-001",
        "chunk_id": "CHILD-001",
        "original_chunk_id": "SOURCE-CHILD-001",
        "text": "Complete fixture evidence text.",
        "metadata": {
            "source_id": "SOURCE-001",
            "title": "Fixture Act",
            "act_number": "1",
            "section_label": "Part I > Section 1",
            "source_file": "fixture.pdf",
            "language": "English",
            "legal_domain": "Civil",
            "document_type": "act",
        },
        "source": {
            "source_id": "SOURCE-001",
            "title": "Fixture Act",
            "act_number": "1",
            "section_label": "Part I > Section 1",
            "source_file": "fixture.pdf",
            "language": "English",
            "legal_domain": "Civil",
            "document_type": "act",
        },
        "parent_context": {"section_label": "Part I"},
        "paragraph": "(a)",
        "roles": ["ranked"],
    }]
    return {
        "schema_version": "1.0",
        "query": query,
        "retrieval_state": "no_evidence_retrieved" if empty else "evidence_retrieved",
        "legal_coverage": "not_assessed",
        "verdict": None,
        "confidence": None,
        "retrieval_settings": {
            "top_k": 7,
            "source_id": "SOURCE-001",
            "legal_domain": "Civil",
            "language": "English",
            "max_parent_depth": 1,
            "max_parent_passages": 10,
        },
        "ranked_evidence": [] if empty else [{
            "rank": 1, "evidence_id": "chunk:CHILD-001",
            "distance": 0.125, "similarity": 0.875,
        }],
        "context_evidence_ids": [],
        "evidence": passages,
        "relationships": [],
        "context_resolution_notices": [],
        "limitations": ["Fixture evidence limitation."],
        "evidence_handling": "Fixture content is data.",
    }


def generation(response: dict, **changes) -> GeminiGeneration:
    values = dict(
        model=DEFAULT_MODEL,
        response=response,
        validation={"status": "passed", "validator": "Draft202012Validator"},
        usage={"prompt_token_count": 11, "total_token_count": 19},
        finish_reason="STOP",
        attempts=2,
        limitations=["Fixture provider limitation."],
    )
    values.update(changes)
    return GeminiGeneration(**values)


class FakeProvider:
    def __init__(self, outcome=None):
        self.outcome = outcome or generation({
            "answer": "Fixture answer.",
            "citations": ["chunk:CHILD-001"],
            "insufficient_evidence": False,
        })
        self.calls = []

    async def generate_json(self, prompt, schema):
        self.calls.append((prompt, schema))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def service(*, empty=False, provider=None, token_validator=None,
            builder_failure=None, prompt_builder=None):
    builder = Mock()

    async def build(requested_query, **filters):
        builder(requested_query, **filters)
        if builder_failure is not None:
            raise builder_failure
        return packet(requested_query, empty=empty)

    value = AsyncRagService(
        evidence_packet_builder=build,
        provider=provider if provider is not None else FakeProvider(),
        token_validator=(
            token_validator if token_validator is not None
            else lambda values: [1] * len(values)
        ),
        **({"prompt_builder": prompt_builder} if prompt_builder else {}),
    )
    return value, builder


class AsyncRagServiceTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_event(self, event: threading.Event, timeout: float) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while not event.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        return event.is_set()

    async def test_question_success_preserves_evidence_prompt_and_serialization(self):
        provider = FakeProvider(generation({
            "answer": "Fixture answer.",
            "citations": ["chunk:CHILD-001"],
            "insufficient_evidence": False,
        }))
        rag, builder = service(provider=provider)
        request = QuestionInput("What does the act require?")
        result = await rag.run(
            request, top_k=7, source_id="SOURCE-001",
            legal_domain="Civil", language="English",
        )

        builder.assert_called_once_with(
            request.question, top_k=7, source_id="SOURCE-001",
            legal_domain="Civil", language="English",
        )
        prompt, schema = provider.calls[0]
        self.assertEqual(prompt["input"], {
            "mode": "question", "question": request.question,
        })
        self.assertEqual(prompt["evidence_data"]["passages"][0]["text"],
                         "Complete fixture evidence text.")
        self.assertNotIn("claim_verdict", schema["properties"])
        self.assertEqual(result.evidence_packet["evidence"][0]["metadata"]["title"],
                         "Fixture Act")
        self.assertEqual(result.response["citations"], ["chunk:CHILD-001"])
        self.assertEqual((result.model, result.usage, result.attempts, result.finish_reason),
                         (DEFAULT_MODEL, {"prompt_token_count": 11, "total_token_count": 19}, 2, "STOP"))
        self.assertEqual(result.limitations, [
            "Fixture evidence limitation.", "Fixture provider limitation.",
        ])

        payload = serialize_rag_result(result).model_dump(mode="json")
        self.assertEqual(payload["mode"], "question")
        self.assertEqual(payload["answer"], "Fixture answer.")
        self.assertEqual(payload["evidence"]["passages"][0]["original_chunk_id"],
                         "SOURCE-CHILD-001")
        self.assertEqual(payload["evidence"]["ranked_evidence"][0]["similarity"], 0.875)
        json.dumps(payload, allow_nan=False)

    async def test_claim_preserves_original_claim_separate_retrieval_query(self):
        tokenized = []
        provider = FakeProvider(generation({
            "claim_verdict": "Misleading",
            "citations": ["chunk:CHILD-001"],
            "insufficient_evidence": False,
            "explanation": "The source narrows the claim.",
        }))
        rag, builder = service(
            provider=provider,
            token_validator=lambda values: tokenized.extend(values),
        )
        claim = ClaimInput("Original claim text.", retrieval_query="Normalized retrieval text.")
        result = await rag.run(
            claim, top_k=7, source_id="SOURCE-001",
            legal_domain="Civil", language="English",
        )
        builder.assert_called_once_with(
            "Normalized retrieval text.", top_k=7, source_id="SOURCE-001",
            legal_domain="Civil", language="English",
        )
        self.assertEqual(tokenized, ["Original claim text.", "Normalized retrieval text."])
        prompt, schema = provider.calls[0]
        self.assertEqual(prompt["input"], {"mode": "claim", "claim": claim.claim})
        self.assertNotIn("Normalized retrieval text.", json.dumps(prompt))
        self.assertEqual(schema["properties"]["claim_verdict"]["enum"],
                         ["True", "False", "Misleading", None])
        self.assertEqual(result.evidence_packet["query"], "Normalized retrieval text.")

    async def test_valid_insufficient_evidence_claim_keeps_null_verdict(self):
        provider = FakeProvider(generation({
            "claim_verdict": None,
            "citations": [],
            "insufficient_evidence": True,
            "explanation": "The available evidence is insufficient.",
        }))
        rag, _ = service(provider=provider)
        result = await rag.run(ClaimInput("The claim."))
        self.assertIsNone(result.response["claim_verdict"])
        self.assertTrue(result.response["insufficient_evidence"])
        self.assertEqual(result.response["citations"], [])

    async def test_empty_evidence_abstains_in_both_modes_without_provider_call(self):
        for request in (QuestionInput("Question?"), ClaimInput("A claim.")):
            with self.subTest(mode=request.mode):
                provider = FakeProvider()
                rag, _ = service(provider=provider, empty=True)
                result = await rag.run(request)
                self.assertEqual(result.response_origin, "application")
                self.assertTrue(result.response["insufficient_evidence"])
                self.assertEqual(result.response["citations"], [])
                self.assertEqual(result.attempts, 0)
                self.assertIsNone(result.model)
                self.assertEqual(provider.calls, [])
                if request.mode == "question":
                    self.assertIsNone(result.response["answer"])
                    self.assertNotIn("claim_verdict", result.response)
                else:
                    self.assertIsNone(result.response["claim_verdict"])
                    self.assertNotIn("answer", result.response)
                serialized = serialize_rag_result(result).model_dump(mode="json")
                self.assertTrue(serialized["insufficient_evidence"])

    async def test_invalid_inputs_fail_before_any_injected_dependency(self):
        token_validator = Mock()
        provider = FakeProvider()
        rag, builder = service(provider=provider, token_validator=token_validator)
        invalid_inputs = (
            QuestionInput(" "),
            ClaimInput(" ", retrieval_query=None),
            ClaimInput("A claim", retrieval_query=" "),
            object(),
        )
        for request in invalid_inputs:
            with self.subTest(request=request), self.assertRaises(RagInputError) as caught:
                await rag.run(request)
            self.assertEqual(caught.exception.category, "input")
        for kwargs in (
            {"top_k": 0}, {"top_k": True}, {"top_k": 51},
            {"source_id": " "}, {"legal_domain": 4},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(RagInputError):
                await rag.run(ClaimInput("Valid claim"), **kwargs)
        builder.assert_not_called()
        token_validator.assert_not_called()
        self.assertEqual(provider.calls, [])

    async def test_token_limit_and_tokenizer_failures_keep_source_categories(self):
        def too_long(values):
            raise EmbeddingInputTooLongError(index=0, token_count=513, limit=512)

        rag, builder = service(token_validator=too_long)
        with self.assertRaises(RagInputTooLongError) as caught:
            await rag.run(QuestionInput("Long question"))
        self.assertEqual((caught.exception.token_count, caught.exception.limit),
                         (513, 512))
        self.assertEqual(caught.exception.category, "input_too_long")
        builder.assert_not_called()

        def unavailable(values):
            raise TokenizerUnavailableError("offline tokenizer missing")

        rag, builder = service(token_validator=unavailable)
        with self.assertRaises(RagPreparationError) as caught:
            await rag.run(ClaimInput("Valid claim"))
        self.assertEqual(caught.exception.category, "preparation")
        builder.assert_not_called()

    async def test_retrieval_and_prompt_failures_are_categorized(self):
        rag, _ = service(builder_failure=RuntimeError("retrieval unavailable"))
        with self.assertRaises(RagRetrievalError) as caught:
            await rag.run(QuestionInput("Question"))
        self.assertEqual(caught.exception.category, "retrieval")

        def bad_prompt(*args):
            raise ValueError("prompt construction failed")

        rag, _ = service(prompt_builder=bad_prompt)
        with self.assertRaises(RagPreparationError) as caught:
            await rag.run(QuestionInput("Question"))
        self.assertEqual(caught.exception.category, "preparation")

    async def test_provider_error_categories_and_request_budget_are_preserved(self):
        for error, expected_type, expected_category in (
            (GeminiAdapterError("provider", failure_category="provider_error"),
             RagProviderError, "provider_error"),
            (GeminiConfigurationError("configuration"),
             RagProviderError, "provider_error"),
            (GeminiRequestPreparationError("prepare", failure_category="request_preparation"),
             RagPreparationError, "request_preparation"),
            (GeminiInvalidResponseError("invalid", failure_category="schema_invalid"),
             RagResponseValidationError, "schema_invalid"),
        ):
            with self.subTest(error=type(error).__name__):
                rag, _ = service(provider=FakeProvider(error))
                with self.assertRaises(expected_type) as caught:
                    await rag.run(QuestionInput("Question"))
                self.assertEqual(caught.exception.category, expected_category)

        too_large = GeminiRequestTooLargeError(
            estimated_input_tokens=17000, project_budget=16384,
            model_input_limit=100000, output_allowance=2048,
        )
        rag, _ = service(provider=FakeProvider(too_large))
        with self.assertRaises(RagRequestTooLargeError) as caught:
            await rag.run(QuestionInput("Question"))
        self.assertEqual(caught.exception.estimated_input_tokens, 17000)
        self.assertEqual(caught.exception.category, "request_too_large")

        async def unexpected(*args):
            raise RuntimeError("provider crashed")

        class Provider:
            generate_json = staticmethod(unexpected)

        rag, _ = service(provider=Provider())
        with self.assertRaises(RagProviderError) as caught:
            await rag.run(QuestionInput("Question"))
        self.assertEqual(caught.exception.category, "provider_error")

    async def test_invalid_citations_and_malformed_generation_are_rejected(self):
        provider = FakeProvider(generation({
            "answer": "Unsupported citation.",
            "citations": ["chunk:NOT-IN-PACKET"],
            "insufficient_evidence": False,
        }))
        rag, _ = service(provider=provider)
        with self.assertRaises(RagResponseValidationError) as caught:
            await rag.run(QuestionInput("Question"))
        self.assertEqual(caught.exception.category, "response_validation")
        self.assertEqual(caught.exception.attempts, 2)

        rag, _ = service(provider=FakeProvider(SimpleNamespace(response={})))
        with self.assertRaises(RagResponseValidationError):
            await rag.run(QuestionInput("Question"))

    async def test_cancellation_propagates_from_retrieval_and_provider(self):
        async def cancelled_builder(query, **kwargs):
            raise asyncio.CancelledError

        rag = AsyncRagService(
            evidence_packet_builder=cancelled_builder,
            provider=FakeProvider(),
            token_validator=lambda values: [1],
        )
        with self.assertRaises(asyncio.CancelledError):
            await rag.run(QuestionInput("Question"))

        class CancelledProvider:
            async def generate_json(self, prompt, schema):
                raise asyncio.CancelledError

        rag, _ = service(provider=CancelledProvider())
        with self.assertRaises(asyncio.CancelledError):
            await rag.run(QuestionInput("Question"))

    async def test_threaded_gemini_boundary_runs_sync_adapter_off_event_loop(self):
        loop_thread = threading.get_ident()
        provider_thread = []

        class SyncAdapter:
            def generate_json(self, prompt, schema):
                provider_thread.append(threading.get_ident())
                return generation({
                    "answer": "Fixture answer.",
                    "citations": ["chunk:CHILD-001"],
                    "insufficient_evidence": False,
                })

        result = await ThreadedGeminiProvider(SyncAdapter).generate_json({}, {})
        self.assertIsInstance(result, GeminiGeneration)
        self.assertEqual(len(provider_thread), 1)
        self.assertNotEqual(provider_thread[0], loop_thread)

    async def test_cancelling_threaded_provider_does_not_return_a_late_rag_result(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        class BlockingAdapter:
            def generate_json(self, prompt, schema):
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test provider was not released")
                finished.set()
                return generation({
                    "answer": "Late result.",
                    "citations": ["chunk:CHILD-001"],
                    "insufficient_evidence": False,
                })

        async def build(query, **kwargs):
            return packet(query)

        rag = AsyncRagService(
            evidence_packet_builder=build,
            provider=ThreadedGeminiProvider(BlockingAdapter),
            token_validator=lambda values: [1] * len(values),
        )
        task = asyncio.create_task(rag.run(QuestionInput("Question")))
        try:
            self.assertTrue(await self._wait_event(started, 2))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.assertTrue(await self._wait_event(finished, 2))

    async def test_worker_adapter_matches_async_analyzer_and_preserves_claim(self):
        class Service:
            def __init__(self):
                self.calls = []

            async def run(self, request, **kwargs):
                self.calls.append((request, kwargs))
                return SimpleNamespace()

        rag = Service()
        analyzer = create_worker_rag_analyzer(
            rag, source_id="SOURCE-001", legal_domain="Civil", language="English",
        )
        result = await analyzer("Claim from worker.", top_k=7)
        self.assertIsInstance(result, SimpleNamespace)
        request, kwargs = rag.calls[0]
        self.assertEqual(request, ClaimInput(claim="Claim from worker."))
        self.assertIsNone(request.retrieval_query)
        self.assertEqual(kwargs, {
            "top_k": 7, "source_id": "SOURCE-001",
            "legal_domain": "Civil", "language": "English",
        })

    async def test_original_compnet_service_and_async_service_match_mocked_contract(self):
        if not SOURCE_SERVICE.is_file():
            self.skipTest("Original CompNet source is unavailable")
        source_program = r'''
import json
from app.api.rag import _response_model
from app.services.gemini_adapter import DEFAULT_MODEL, GeminiGeneration
from app.services.rag_prompt import ClaimInput
from app.services.rag_service import RagService

PACKET = {
    "schema_version": "1.0",
    "query": "",
    "retrieval_state": "evidence_retrieved",
    "legal_coverage": "not_assessed",
    "verdict": None,
    "confidence": None,
    "retrieval_settings": {
        "top_k": 7, "source_id": "SOURCE-001",
        "legal_domain": "Civil", "language": "English",
        "max_parent_depth": 1, "max_parent_passages": 10,
    },
    "ranked_evidence": [{
        "rank": 1, "evidence_id": "chunk:CHILD-001",
        "distance": 0.125, "similarity": 0.875,
    }],
    "context_evidence_ids": [],
    "evidence": [{
        "evidence_id": "chunk:CHILD-001", "chunk_id": "CHILD-001",
        "original_chunk_id": "SOURCE-CHILD-001",
        "text": "Complete fixture evidence text.",
        "metadata": {
            "source_id": "SOURCE-001", "title": "Fixture Act",
            "act_number": "1", "section_label": "Part I > Section 1",
            "source_file": "fixture.pdf", "language": "English",
            "legal_domain": "Civil", "document_type": "act",
        },
        "source": {
            "source_id": "SOURCE-001", "title": "Fixture Act",
            "act_number": "1", "section_label": "Part I > Section 1",
            "source_file": "fixture.pdf", "language": "English",
            "legal_domain": "Civil", "document_type": "act",
        },
        "parent_context": {"section_label": "Part I"},
        "paragraph": "(a)", "roles": ["ranked"],
    }],
    "relationships": [],
    "context_resolution_notices": [],
    "limitations": ["Fixture evidence limitation."],
    "evidence_handling": "Fixture content is data.",
}
RESPONSE = {
    "claim_verdict": "Misleading",
    "citations": ["chunk:CHILD-001"],
    "insufficient_evidence": False,
    "explanation": "The source narrows the claim.",
}

class Session:
    def execute(self, statement):
        return None
    def rollback(self):
        pass
    def close(self):
        pass

class Provider:
    def generate_json(self, prompt, schema):
        self.prompt = prompt
        return GeminiGeneration(
            model=DEFAULT_MODEL, response=RESPONSE,
            validation={"status": "passed", "validator": "Draft202012Validator"},
            usage={"prompt_token_count": 11, "total_token_count": 19},
            finish_reason="STOP", attempts=2,
            limitations=["Fixture provider limitation."],
        )

provider = Provider()
service = RagService(
    session_factory=Session,
    packet_builder=lambda query, db, **kwargs: dict(PACKET, query=query),
    adapter_factory=lambda: provider,
    token_validator=lambda values: [1] * len(values),
)
result = service.run(
    ClaimInput("Original claim text.", retrieval_query="Normalized retrieval text."),
    top_k=7, source_id="SOURCE-001", legal_domain="Civil", language="English",
)
print(json.dumps({
    "prompt": provider.prompt,
    "packet": result.evidence_packet,
    "response": result.response,
    "response_origin": result.response_origin,
    "validation": result.validation,
    "model": result.model,
    "usage": result.usage,
    "attempts": result.attempts,
    "finish_reason": result.finish_reason,
    "limitations": result.limitations,
    "serialized": _response_model(result).model_dump(mode="json"),
}, ensure_ascii=False, sort_keys=True))
'''
        env = {
            "PATH": os.defpath,
            "PYTHONPATH": str(SOURCE_BACKEND),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        completed = subprocess.run(
            [os.fspath(Path(os.sys.executable)), "-c", source_program],
            cwd=SOURCE_BACKEND,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        source = json.loads(completed.stdout)

        provider = FakeProvider(generation({
            "claim_verdict": "Misleading",
            "citations": ["chunk:CHILD-001"],
            "insufficient_evidence": False,
            "explanation": "The source narrows the claim.",
        }))
        async def build(query, **kwargs):
            return packet(query)

        rag = AsyncRagService(
            evidence_packet_builder=build,
            provider=provider,
            token_validator=lambda values: [1] * len(values),
        )
        result = await rag.run(
            ClaimInput("Original claim text.", retrieval_query="Normalized retrieval text."),
            top_k=7, source_id="SOURCE-001", legal_domain="Civil", language="English",
        )
        target = {
            "prompt": provider.calls[0][0],
            "packet": result.evidence_packet,
            "response": result.response,
            "response_origin": result.response_origin,
            "validation": result.validation,
            "model": result.model,
            "usage": result.usage,
            "attempts": result.attempts,
            "finish_reason": result.finish_reason,
            "limitations": result.limitations,
            "serialized": serialize_rag_result(result).model_dump(mode="json"),
        }
        self.assertEqual(target, source)


if __name__ == "__main__":
    unittest.main()
