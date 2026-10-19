"""Database-independent async RAG orchestration for C3 question and claim modes."""
from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from jsonschema import Draft202012Validator

from .agents.gemini_adapter import (
    DEFAULT_MODEL,
    GeminiAdapterError,
    GeminiInvalidResponseError,
    GeminiRequestPreparationError,
    GeminiRequestTooLargeError,
)
from .pipeline.claim_input_tokens import (
    EmbeddingInputTooLongError,
    TokenizerUnavailableError,
    validate_claim_input_tokens,
)
from .pipeline.multilingual_claim_adapter import MultilingualClaimAdapter
from .prompts.rag_prompt import (
    ClaimInput,
    Input,
    QuestionInput,
    build_rag_prompt,
)
from .rag_contract import (
    RagInputError,
    RagInputTooLongError,
    RagPrepared,
    RagPreparationError,
    RagProviderError,
    RagRequestTooLargeError,
    RagResponseValidationError,
    RagResult,
    RagRetrievalError,
)


class AsyncEvidencePacketBuilder(Protocol):
    def __call__(
        self,
        query: str,
        *,
        top_k: int,
        source_id: str | None,
        legal_domain: str | None,
        language: str | None,
    ) -> Awaitable[dict[str, Any]]:
        """Return an awaitable C3 evidence packet from injected dependencies."""


class AsyncGeminiProvider(Protocol):
    async def generate_json(
        self, prompt: dict[str, Any], schema: dict[str, Any],
    ) -> Any:
        """Generate and validate one structured response."""


class SyncGeminiAdapter(Protocol):
    def generate_json(
        self, prompt: dict[str, Any], schema: dict[str, Any],
    ) -> Any:
        """Generate JSON synchronously through the existing C3 adapter."""


class ThreadedGeminiProvider:
    """Async boundary for the existing synchronous C3 GeminiAdapter.

    Cancellation of the awaiting coroutine cannot terminate a provider call
    already running in the worker thread. This adapter has no persistence
    capability; worker result writes remain separately ownership-fenced.
    """

    def __init__(self, adapter_factory: Callable[[], SyncGeminiAdapter]) -> None:
        if not callable(adapter_factory):
            raise TypeError("adapter_factory must be callable")
        self._adapter_factory = adapter_factory

    async def generate_json(
        self, prompt: dict[str, Any], schema: dict[str, Any],
    ) -> Any:
        return await _run_sync_daemon(self._generate_json, prompt, schema)

    def _generate_json(
        self, prompt: dict[str, Any], schema: dict[str, Any],
    ) -> Any:
        return self._adapter_factory().generate_json(prompt, schema)


async def _run_sync_daemon(callable_: Callable[..., Any], *args: Any) -> Any:
    """Run a synchronous boundary without retaining a default executor thread.

    Completion is observed through a threading event instead of a
    ``call_soon_threadsafe`` callback.  Some supported Python/AnyIO runtime
    combinations can leave that callback queued while the loop is asleep in
    its selector, which makes an otherwise finished provider await forever.
    """
    # Validate that this boundary is called from an active event loop.  The
    # completion event below is deliberately polled so loop wakeups do not
    # depend on cross-thread callback delivery.
    asyncio.get_running_loop()
    value: Any = None
    error: BaseException | None = None
    finished = threading.Event()

    def complete() -> None:
        nonlocal value, error
        try:
            value = callable_(*args)
        except BaseException as exc:
            error = exc
        finally:
            finished.set()

    thread = threading.Thread(
        target=complete, name="c3-rag-sync-boundary", daemon=True,
    )
    thread.start()
    try:
        while not finished.is_set():
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        # A synchronous provider cannot be force-stopped safely.  Leave only
        # the bounded daemon call in flight and preserve task cancellation.
        raise
    thread.join(timeout=0.1)
    if error is not None:
        raise error
    return value


def _validate_input(input_data: Input) -> tuple[Literal["question", "claim"], str]:
    if isinstance(input_data, QuestionInput):
        if (input_data.mode != "question" or not isinstance(input_data.question, str)
                or not input_data.question.strip()):
            raise RagInputError(
                "question input must explicitly use question mode with nonblank text",
                category="input",
            )
        return "question", input_data.question
    if isinstance(input_data, ClaimInput):
        if (input_data.mode != "claim" or not isinstance(input_data.claim, str)
                or not input_data.claim.strip()):
            raise RagInputError(
                "claim input must explicitly use claim mode with nonblank text",
                category="input",
            )
        return "claim", input_data.claim
    raise RagInputError("input must be QuestionInput or ClaimInput", category="input")


def _retrieval_query(input_data: Input, request_text: str) -> str:
    if not isinstance(input_data, ClaimInput) or input_data.retrieval_query is None:
        return request_text
    query = input_data.retrieval_query
    if not isinstance(query, str) or not query.strip():
        raise RagInputError(
            "retrieval_query must be a nonblank string when provided",
            category="input",
        )
    return query


def _validate_filters(
    top_k: int,
    source_id: str | None,
    legal_domain: str | None,
    language: str | None,
) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise RagInputError("top_k must be an integer between 1 and 50", category="input")
    for name, value in (
        ("source_id", source_id), ("legal_domain", legal_domain), ("language", language),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise RagInputError(
                f"{name} must be a nonblank string when provided",
                category="input",
            )


class AsyncRagService:
    """Orchestrate RAG while leaving database, embedding, and provider ownership injected."""

    def __init__(
        self,
        *,
        evidence_packet_builder: AsyncEvidencePacketBuilder,
        provider: AsyncGeminiProvider,
        prompt_builder: Callable[[Input, dict[str, Any]], dict[str, Any]] = build_rag_prompt,
        token_validator: Callable[[list[str]], list[int]] = validate_claim_input_tokens,
        model: str = DEFAULT_MODEL,
        multilingual_adapter: MultilingualClaimAdapter | None = None,
    ) -> None:
        if model != DEFAULT_MODEL:
            raise RagInputError(f"model must remain {DEFAULT_MODEL}", category="input")
        if not callable(evidence_packet_builder):
            raise TypeError("evidence_packet_builder must be callable")
        if not callable(getattr(provider, "generate_json", None)):
            raise TypeError("provider must implement async generate_json")
        if multilingual_adapter is not None and not isinstance(multilingual_adapter, MultilingualClaimAdapter):
            raise TypeError("multilingual_adapter must be a MultilingualClaimAdapter or None")
        self._evidence_packet_builder = evidence_packet_builder
        self._provider = provider
        self._prompt_builder = prompt_builder
        self._token_validator = token_validator
        self._multilingual_adapter = multilingual_adapter
        self.model = model

    async def _prepare_input(self, input_data: Input) -> Input:
        if self._multilingual_adapter is not None:
            return await self._multilingual_adapter.aprepare_input(input_data)
        return input_data

    async def prepare(
        self,
        input_data: Input,
        *,
        top_k: int = 5,
        source_id: str | None = None,
        legal_domain: str | None = None,
        language: str | None = None,
    ) -> RagPrepared:
        input_data = await self._prepare_input(input_data)
        mode, request_text = _validate_input(input_data)
        retrieval_query = _retrieval_query(input_data, request_text)
        _validate_filters(top_k, source_id, legal_domain, language)

        tokenized_texts = [request_text]
        if retrieval_query != request_text:
            tokenized_texts.append(retrieval_query)
        try:
            # Token validation is a local, bounded preflight operation. Keep it
            # in the coroutine so cancellation/error categories are deterministic
            # and no executor lifecycle is introduced into unittest loops.
            self._token_validator(tokenized_texts)
        except EmbeddingInputTooLongError as exc:
            raise RagInputTooLongError(
                token_count=exc.token_count, limit=exc.limit,
            ) from exc
        except TokenizerUnavailableError as exc:
            raise RagPreparationError(
                "Local claim tokenizer is unavailable.", category="preparation",
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RagPreparationError(
                f"RAG input preparation failed: {type(exc).__name__}: {exc}",
                category="preparation",
            ) from exc

        try:
            packet_result = self._evidence_packet_builder(
                retrieval_query,
                top_k=top_k,
                source_id=source_id,
                legal_domain=legal_domain,
                language=language,
            )
            if not inspect.isawaitable(packet_result):
                raise TypeError("evidence_packet_builder must return an awaitable packet")
            packet = await packet_result
            if not isinstance(packet, dict):
                raise TypeError("evidence_packet_builder must return a packet object")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RagRetrievalError(
                f"RAG evidence retrieval failed: {type(exc).__name__}: {exc}",
                category="retrieval",
            ) from exc

        try:
            prompt = self._prompt_builder(input_data, packet)
            if not isinstance(prompt, dict):
                raise TypeError("RAG prompt builder must return an object")
            schema = prompt["response_contract"]
            Draft202012Validator.check_schema(schema)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RagPreparationError(
                f"RAG prompt preparation failed: {type(exc).__name__}: {exc}",
                category="preparation",
            ) from exc

        # Database resources and transaction state belong to the injected
        # evidence builder's caller; this service must not claim cleanup status.
        return RagPrepared(
            mode=mode,
            evidence_packet=packet,
            prompt=prompt,
            schema=schema,
            transaction_status={},
        )

    async def run(
        self,
        input_data: Input,
        *,
        top_k: int = 5,
        source_id: str | None = None,
        legal_domain: str | None = None,
        language: str | None = None,
    ) -> RagResult:
        prepared = await self.prepare(
            input_data,
            top_k=top_k,
            source_id=source_id,
            legal_domain=legal_domain,
            language=language,
        )
        mode = prepared.mode
        packet = prepared.evidence_packet
        prompt = prepared.prompt
        schema = prepared.schema

        if not prompt.get("evidence_data", {}).get("passages", []):
            if mode == "question":
                response = {
                    "answer": None,
                    "citations": [],
                    "insufficient_evidence": True,
                    "explanation": "No evidence was retrieved for this request.",
                }
            else:
                response = {
                    "claim_verdict": None,
                    "citations": [],
                    "insufficient_evidence": True,
                    "explanation": "No evidence was retrieved for this request.",
                }
            errors = list(Draft202012Validator(schema).iter_errors(response))
            if errors:
                raise RagResponseValidationError(
                    "Application abstention failed local schema validation: "
                    f"{errors[0].message}",
                    category="response_validation",
                )
            return RagResult(
                mode=mode,
                evidence_packet=packet,
                response=response,
                response_origin="application",
                validation={
                    "status": "passed",
                    "validator": "Draft202012Validator",
                    "origin": "application",
                },
                model=None,
                usage=None,
                attempts=0,
                finish_reason=None,
                limitations=list(packet.get("limitations", [])) + [
                    "Application-generated abstention; this does not imply that relevant law does not exist.",
                    "No hosted model request was made because the evidence packet was empty.",
                ],
                transaction_status=prepared.transaction_status,
            )

        try:
            generated = await self._provider.generate_json(prompt, schema)
        except asyncio.CancelledError:
            raise
        except GeminiRequestTooLargeError as exc:
            raise RagRequestTooLargeError(
                estimated_input_tokens=exc.estimated_input_tokens,
                project_budget=exc.project_budget,
                model_input_limit=exc.model_input_limit,
                output_allowance=exc.output_allowance,
            ) from exc
        except GeminiRequestPreparationError as exc:
            raise RagPreparationError(
                str(exc), category=exc.failure_category, attempts=exc.attempts,
            ) from exc
        except GeminiInvalidResponseError as exc:
            raise RagResponseValidationError(
                str(exc), category=exc.failure_category, attempts=exc.attempts,
            ) from exc
        except GeminiAdapterError as exc:
            raise RagProviderError(
                str(exc), category=exc.failure_category, attempts=exc.attempts,
            ) from exc
        except Exception as exc:
            raise RagProviderError(
                f"RAG provider call failed: {type(exc).__name__}: {exc}",
                category="provider_error",
            ) from exc

        try:
            response = generated.response
            errors = list(Draft202012Validator(schema).iter_errors(response))
            if errors:
                raise RagResponseValidationError(
                    f"Provider response failed local schema validation: {errors[0].message}",
                    category="response_validation",
                    attempts=generated.attempts,
                )
            if not isinstance(generated.validation, dict):
                raise TypeError("provider validation metadata must be an object")
            if generated.model is not None and not isinstance(generated.model, str):
                raise TypeError("provider model identity must be text or null")
            if generated.usage is not None and not isinstance(generated.usage, dict):
                raise TypeError("provider usage metadata must be an object or null")
            if not isinstance(generated.attempts, int) or isinstance(generated.attempts, bool):
                raise TypeError("provider attempts must be an integer")
            if generated.finish_reason is not None and not isinstance(generated.finish_reason, str):
                raise TypeError("provider finish reason must be text or null")
            if not isinstance(generated.limitations, list) or not all(
                isinstance(item, str) for item in generated.limitations
            ):
                raise TypeError("provider limitations must be a list of strings")
        except RagResponseValidationError:
            raise
        except Exception as exc:
            raise RagResponseValidationError(
                f"Provider response failed local validation: {type(exc).__name__}: {exc}",
                category="response_validation",
                attempts=getattr(generated, "attempts", 0),
            ) from exc

        return RagResult(
            mode=mode,
            evidence_packet=packet,
            response=response,
            response_origin="model",
            validation=generated.validation,
            model=generated.model,
            usage=generated.usage,
            attempts=generated.attempts,
            finish_reason=generated.finish_reason,
            limitations=list(packet.get("limitations", [])) + list(generated.limitations),
            transaction_status=prepared.transaction_status,
        )


__all__ = [
    "AsyncEvidencePacketBuilder",
    "AsyncGeminiProvider",
    "AsyncRagService",
    "ThreadedGeminiProvider",
]
