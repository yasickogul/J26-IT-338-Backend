"""Lazy production composition for the explicitly invoked C3 claim worker.

Runtime prerequisites are an existing async session factory, the applied C3
claim and legal-corpus PostgreSQL schemas, cached LaBSE weights, and ``GEMINI_API_KEY``
plus ``google-genai`` when evidence requires hosted analysis. Composition
performs no database, model, or provider operation.
"""
from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .agents.gemini_adapter import DEFAULT_MODEL, GeminiAdapter
from .claim_worker import AsyncAnalyzer
from .pipeline.c3_corpus_evidence import build_c3_evidence_packet
from .pipeline.claim_input_tokens import validate_claim_input_tokens
from .pipeline.labse_query_embedding import EXPECTED_DIMENSION, MODEL_NAME, embed_text
from .pipeline.multilingual_claim_adapter import MultilingualClaimAdapter
from .pipeline.retrieval_adapter import AsyncQueryEmbedder
from .rag_service_async import AsyncRagService, ThreadedGeminiProvider
from .worker_rag_adapter import create_worker_rag_analyzer

AsyncSessionFactory = Callable[[], AsyncSession | Awaitable[AsyncSession]]
_labse_encoder: Any | None = None
_labse_encoder_lock = Lock()


class WorkerDependencyConfigurationError(RuntimeError):
    """The C3 worker cannot be composed from configured runtime dependencies."""


@dataclass(frozen=True)
class WorkerRuntimeConfig:
    top_k: int = 5
    source_id: str | None = None
    legal_domain: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if type(self.top_k) is not int or not 1 <= self.top_k <= 50:
            raise ValueError("top_k must be an integer between 1 and 50")
        for name in ("source_id", "legal_domain", "language"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonblank string when provided")


@dataclass(frozen=True)
class WorkerDependencies:
    session_factory: AsyncSessionFactory
    query_embedder: AsyncQueryEmbedder
    evidence_packet_builder: Callable[..., Awaitable[dict[str, Any]]]
    gemini_provider: ThreadedGeminiProvider
    rag_service: AsyncRagService
    analyzer: AsyncAnalyzer
    config: WorkerRuntimeConfig


def _configured_session_factory() -> AsyncSessionFactory:
    from app.core import database

    session_factory = database.SessionLocal
    if session_factory is None:
        raise WorkerDependencyConfigurationError(
            "The C3 worker requires the existing DATABASE_URL-backed session factory."
        )
    return session_factory


def _local_labse_embedder() -> AsyncQueryEmbedder:
    async def embed_query(query: str):
        asyncio.get_running_loop()
        value: Any = None
        error: BaseException | None = None
        finished = threading.Event()

        def work() -> None:
            nonlocal value, error
            try:
                value = _embed_with_cached_labse(query)
            except BaseException as exc:  # noqa: BLE001 - transfer thread outcome
                error = exc
            finally:
                finished.set()

        thread = threading.Thread(target=work, name="c3-labse-embedder", daemon=True)
        thread.start()
        while not finished.is_set():
            await asyncio.sleep(0.01)
        thread.join(timeout=0.1)
        if error is not None:
            raise error
        return value

    return embed_query


def _get_cached_labse_encoder() -> Any:
    """Load the required model only from the local Hugging Face cache."""
    global _labse_encoder
    if _labse_encoder is not None:
        return _labse_encoder
    with _labse_encoder_lock:
        if _labse_encoder is not None:
            return _labse_encoder
        try:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer

            snapshot = snapshot_download(MODEL_NAME, local_files_only=True)
            encoder = SentenceTransformer(str(snapshot), device="cpu")
        except Exception as exc:
            raise WorkerDependencyConfigurationError(
                "Cached sentence-transformers/LaBSE weights are unavailable; "
                "provision the approved model locally before running the C3 worker."
            ) from exc

        dimension_getter = getattr(
            encoder,
            "get_embedding_dimension",
            getattr(encoder, "get_sentence_embedding_dimension", None),
        )
        if dimension_getter is None or dimension_getter() != EXPECTED_DIMENSION:
            raise WorkerDependencyConfigurationError(
                "Cached sentence-transformers/LaBSE model has an incompatible embedding dimension."
            )
        _labse_encoder = encoder
        return encoder


def _embed_with_cached_labse(query: str):
    return embed_text(query, encoder=_get_cached_labse_encoder())


def create_worker_dependencies(
    *,
    session_factory: AsyncSessionFactory,
    config: WorkerRuntimeConfig | None = None,
    query_embedder: AsyncQueryEmbedder | None = None,
    gemini_adapter_factory: Callable[[], GeminiAdapter] | None = None,
    multilingual_adapter: MultilingualClaimAdapter | None = None,
    token_validator: Callable[[list[str]], list[int]] = validate_claim_input_tokens,
) -> WorkerDependencies:
    """Compose existing C3 evidence, RAG, Gemini, and worker adapters.

    All expensive/runtime operations remain behind async invocation. The
    evidence builder owns one short-lived read-only session per analysis;
    worker lifecycle and lease-renewal sessions remain owned by claim_worker.
    The tokenizer validator can be injected for controlled offline execution.
    """
    if config is None:
        config = WorkerRuntimeConfig()
    if not callable(session_factory):
        raise TypeError("session_factory must be callable")
    if not isinstance(config, WorkerRuntimeConfig):
        raise TypeError("config must be WorkerRuntimeConfig")
    if not callable(token_validator):
        raise TypeError("token_validator must be callable")
    if multilingual_adapter is not None and not isinstance(multilingual_adapter, MultilingualClaimAdapter):
        raise TypeError("multilingual_adapter must be a MultilingualClaimAdapter or None")

    embedder = (
        query_embedder if query_embedder is not None else _local_labse_embedder()
    )

    async def evidence_packet_builder(
        query: str,
        *,
        top_k: int,
        source_id: str | None,
        legal_domain: str | None,
        language: str | None,
    ) -> dict[str, Any]:
        session = session_factory()
        if inspect.isawaitable(session):
            session = await session
        packet: dict[str, Any] | None = None
        failure: BaseException | None = None
        try:
            if not all(
                callable(getattr(session, name, None))
                for name in ("execute", "rollback", "close")
            ):
                raise WorkerDependencyConfigurationError(
                    "The C3 worker session factory returned an invalid async session."
                )
            await session.execute(text("SET TRANSACTION READ ONLY"))
            packet = await build_c3_evidence_packet(
                query,
                session=session,
                embedder=embedder,
                query_model=MODEL_NAME,
                top_k=top_k,
                source_id=source_id,
                legal_domain=legal_domain,
                language=language,
            )
        except BaseException as exc:  # noqa: BLE001 - cleanup must follow any failure
            failure = exc

        for cleanup_name in ("rollback", "close"):
            cleanup_method = getattr(session, cleanup_name, None)
            if not callable(cleanup_method):
                continue
            try:
                cleanup = cleanup_method()
                if inspect.isawaitable(cleanup):
                    await cleanup
            except BaseException as exc:  # noqa: BLE001 - preserve primary failure
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(f"{cleanup_name} after evidence retrieval also failed")
        if failure is not None:
            raise failure
        if packet is None:
            raise RuntimeError("Evidence packet builder returned no packet.")
        return packet

    provider = ThreadedGeminiProvider(
        gemini_adapter_factory
        if gemini_adapter_factory is not None
        else (lambda: GeminiAdapter(model=DEFAULT_MODEL)),
    )
    rag_service = AsyncRagService(
        evidence_packet_builder=evidence_packet_builder,
        provider=provider,
        token_validator=token_validator,
        model=DEFAULT_MODEL,
        multilingual_adapter=multilingual_adapter,
    )
    analyzer = create_worker_rag_analyzer(
        rag_service,
        source_id=config.source_id,
        legal_domain=config.legal_domain,
        language=config.language,
    )
    return WorkerDependencies(
        session_factory=session_factory,
        query_embedder=embedder,
        evidence_packet_builder=evidence_packet_builder,
        gemini_provider=provider,
        rag_service=rag_service,
        analyzer=analyzer,
        config=config,
    )


def create_configured_worker_dependencies(
    *,
    config: WorkerRuntimeConfig | None = None,
    query_embedder: AsyncQueryEmbedder | None = None,
    gemini_adapter_factory: Callable[[], GeminiAdapter] | None = None,
    multilingual_adapter: MultilingualClaimAdapter | None = None,
    token_validator: Callable[[list[str]], list[int]] = validate_claim_input_tokens,
) -> WorkerDependencies:
    """Use the group's existing database sessionmaker without connecting."""
    return create_worker_dependencies(
        session_factory=_configured_session_factory(),
        config=config,
        query_embedder=query_embedder,
        gemini_adapter_factory=gemini_adapter_factory,
        multilingual_adapter=multilingual_adapter,
        token_validator=token_validator,
    )


__all__ = [
    "WorkerDependencies",
    "WorkerDependencyConfigurationError",
    "WorkerRuntimeConfig",
    "create_configured_worker_dependencies",
    "create_worker_dependencies",
]
