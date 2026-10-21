"""Async one-shot orchestration for a leased C3 claim.

Repository operations are deliberately given a fresh session for every
lifecycle action.  Analysis is injected so this module does not provide a
default RAG implementation or call a blocking provider on the event loop.
"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from . import repository
from .claim_lifecycle_contract import (
    FAILURE_CATEGORIES,
    ClaimedTextClaim,
    ClaimStateError,
    ClaimStorageError,
)
from .claim_worker_contract import ClaimExtractionWorkerError, WorkerOutcome
from .config import claim_extraction_enabled, llm_provider
from .pipeline.claim_extraction import (
    ClaimExtractionProvider,
    ClaimExtractionResult,
    GeminiClaimExtractionProvider,
    extract_legal_claim,
)
from .pipeline.qwen_claim_extraction import QwenClaimExtractionProvider
from .rag_contract import RagResult, RagServiceError
from .rag_response import serialize_rag_result

AsyncSessionFactory = Callable[[], AsyncSession | Awaitable[AsyncSession]]
AsyncAnalyzer = Callable[[str], Awaitable[RagResult]]
ExtractorFactory = Callable[[], ClaimExtractionProvider]


def get_claim_extraction_enabled() -> bool:
    """Read the original opt-in extraction feature flag."""
    return claim_extraction_enabled()


def _extraction_provenance(result: ClaimExtractionResult) -> dict[str, str | None]:
    return {
        "status": result.status,
        "method": result.method,
        "failure_reason": result.failure_reason,
        "input_source": "cleaned_text",
        "stored_claim_field": "claims.extracted_claim",
    }


async def _new_session(factory: AsyncSessionFactory) -> AsyncSession:
    value = factory()
    if inspect.isawaitable(value):
        value = await value
    return value


async def _close_session(session: AsyncSession) -> None:
    result = session.close()
    if inspect.isawaitable(result):
        await result


async def _recover(factory: AsyncSessionFactory) -> list[Any]:
    session = await _new_session(factory)
    try:
        return await repository.recover_expired_claims(session=session)
    finally:
        await _close_session(session)


async def _acquire(factory: AsyncSessionFactory, lease_seconds: int) -> ClaimedTextClaim | None:
    session = await _new_session(factory)
    try:
        return await repository.claim_next_queued_text_claim(
            session=session, lease_seconds=lease_seconds,
        )
    finally:
        await _close_session(session)


async def _renew(
    claim: ClaimedTextClaim,
    factory: AsyncSessionFactory,
    lease_seconds: int,
    stop: asyncio.Event,
    lost: asyncio.Event,
) -> None:
    interval = min(30.0, lease_seconds / 3)
    while True:
        stop_wait = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait({stop_wait}, timeout=interval)
            if done:
                return
        finally:
            if not stop_wait.done():
                stop_wait.cancel()
                with suppress(asyncio.CancelledError):
                    await stop_wait
        try:
            session = await _new_session(factory)
            try:
                await repository.renew_claim_lease(
                    claim.claim_id, claim.worker_token,
                    session=session, lease_seconds=lease_seconds,
                )
            finally:
                await _close_session(session)
        except ClaimStateError:
            lost.set()
            return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001, S112 - retry transient renewal next interval
            # A transient renewal failure does not transfer ownership.  The
            # next interval gets another chance, matching the source worker.
            continue


async def _write_extracted(
    factory: AsyncSessionFactory, claim: ClaimedTextClaim, extracted: str,
) -> None:
    session = await _new_session(factory)
    try:
        await repository.record_extracted_claim(
            claim.claim_id, extracted, claim.worker_token, session=session,
        )
    finally:
        await _close_session(session)


async def _write_result(
    factory: AsyncSessionFactory, claim: ClaimedTextClaim,
    payload: dict[str, Any], top_k: int,
) -> Any:
    session = await _new_session(factory)
    try:
        return await repository.record_completed_result(
            claim.claim_id, payload, retrieval_top_k=top_k,
            worker_token=claim.worker_token, session=session,
        )
    finally:
        await _close_session(session)


async def _write_failure(
    factory: AsyncSessionFactory, claim: ClaimedTextClaim, category: str,
) -> None:
    session = await _new_session(factory)
    try:
        await repository.record_claim_failure(
            claim.claim_id, category, worker_token=claim.worker_token, session=session,
        )
    finally:
        await _close_session(session)


async def _run_extraction(
    text: str, extractor_factory: ExtractorFactory | None,
) -> ClaimExtractionResult:
    # Both provider construction and extraction are synchronous provider
    # boundaries; isolate them from the asyncio event loop.
    factory = extractor_factory or (
        QwenClaimExtractionProvider if llm_provider() == "qwen"
        else GeminiClaimExtractionProvider
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="claim-extraction")
    try:
        provider_future = executor.submit(factory)
        while not provider_future.done():
            await asyncio.sleep(0.01)
        provider = provider_future.result()
        extraction_future = executor.submit(
            partial(extract_legal_claim, text, provider=provider),
        )
        while not extraction_future.done():
            await asyncio.sleep(0.01)
        return extraction_future.result()
    finally:
        # A running synchronous provider cannot be forcefully interrupted;
        # do not block cancellation on executor shutdown.
        executor.shutdown(wait=False, cancel_futures=True)


def _extraction_error(result: ClaimExtractionResult) -> ClaimExtractionWorkerError | None:
    if result.status == "extracted" and result.extracted_claim is not None:
        return None
    if result.status in {"multiple_claims", "ambiguous", "no_legal_assertion"}:
        category = "input"
    elif result.failure_reason == "invalid_provider_response":
        category = "response_validation"
    else:
        category = "provider"
    return ClaimExtractionWorkerError(
        result.failure_reason or result.status, failure_category=category,
    )


async def process_one_claim(
    *,
    analysis: AsyncAnalyzer,
    session_factory: AsyncSessionFactory,
    lease_seconds: int = 3600,
    top_k: int = 5,
    extraction_enabled: bool | None = None,
    extractor_factory: ExtractorFactory | None = None,
) -> WorkerOutcome | None:
    """Recover and process at most one queued claim asynchronously.

    ``analysis`` must be an async callable accepting claim text and a
    ``top_k`` keyword argument. Production composition injects the C3-owned
    corpus RAG adapter; tests may inject a deterministic offline adapter.
    """
    if extraction_enabled is None:
        extraction_enabled = get_claim_extraction_enabled()
    elif type(extraction_enabled) is not bool:
        raise TypeError("extraction_enabled must be a boolean or None")
    if type(top_k) is not int or not 1 <= top_k <= 50:
        raise ValueError("top_k must be an integer from 1 to 50")
    if not callable(analysis):
        raise TypeError("analysis must be an async callable")

    await _recover(session_factory)
    claim = await _acquire(session_factory, lease_seconds)
    if claim is None:
        return None

    stop = asyncio.Event()
    ownership_lost = asyncio.Event()
    renewal = asyncio.create_task(
        _renew(claim, session_factory, lease_seconds, stop, ownership_lost),
        name="claim-lease-renewal",
    )
    try:
        processing_text = (
            claim.cleaned_text if claim.cleaned_text is not None else claim.raw_input_text
        )
        extraction_result: ClaimExtractionResult | None = None
        if extraction_enabled:
            extraction_result = await _run_extraction(processing_text, extractor_factory)
            extraction_error = _extraction_error(extraction_result)
            if extraction_error is not None:
                raise extraction_error
            await _write_extracted(
                session_factory, claim, extraction_result.extracted_claim,  # type: ignore[arg-type]
            )
            processing_text = extraction_result.extracted_claim

        generated = analysis(processing_text, top_k=top_k)
        if not inspect.isawaitable(generated):
            raise TypeError("analysis must return an awaitable RagResult")
        result = await generated
        payload = serialize_rag_result(result).model_dump(mode="json")
        if extraction_result is not None:
            payload["claim_extraction"] = _extraction_provenance(extraction_result)
        await _write_result(session_factory, claim, payload, top_k)
        return WorkerOutcome(claim.claim_id, "completed")
    except asyncio.CancelledError:
        raise
    except ClaimStateError:
        await _recover(session_factory)
        return WorkerOutcome(claim.claim_id, "ownership_lost")
    except Exception as exc:  # noqa: BLE001 - worker maps failures to safe categories
        if isinstance(exc, ClaimStorageError):
            category = "result_storage"
        elif isinstance(exc, ClaimExtractionWorkerError):
            category = exc.failure_category
        elif isinstance(exc, RagServiceError) and exc.category in FAILURE_CATEGORIES:
            category = exc.category
        else:
            category = "service"
        try:
            await _write_failure(session_factory, claim, category)
        except ClaimStateError:
            await _recover(session_factory)
            return WorkerOutcome(claim.claim_id, "ownership_lost")
        return WorkerOutcome(claim.claim_id, "failed")
    finally:
        stop.set()
        # Let the renewal loop observe the stop event.  Shielding prevents a
        # cancellation of the processing task from leaving a heartbeat task
        # half-cancelled; force-cancel only if it does not finish promptly.
        try:
            await asyncio.wait_for(asyncio.shield(renewal), timeout=2.0)
        except TimeoutError:
            renewal.cancel()
            with suppress(asyncio.CancelledError):
                await renewal


__all__ = [
    "AsyncAnalyzer", "AsyncSessionFactory", "ClaimExtractionWorkerError",
    "WorkerOutcome", "get_claim_extraction_enabled", "process_one_claim",
]
