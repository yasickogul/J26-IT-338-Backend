"""Adapt the async RAG service to the existing async claim-worker analyzer API."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from .pipeline.multilingual_claim_adapter import MultilingualClaimAdapter
from .prompts.rag_prompt import ClaimInput
from .rag_contract import RagResult


class AsyncRagService(Protocol):
    async def run(
        self,
        input_data: ClaimInput,
        *,
        top_k: int,
        source_id: str | None,
        legal_domain: str | None,
        language: str | None,
    ) -> RagResult:
        """Analyze one explicit claim using injected C3 dependencies."""


def create_worker_rag_analyzer(
    service: AsyncRagService,
    *,
    source_id: str | None = None,
    legal_domain: str | None = None,
    language: str | None = None,
    multilingual_adapter: MultilingualClaimAdapter | None = None,
) -> Callable[..., Awaitable[RagResult]]:
    """Bind service/filter dependencies to ``AsyncAnalyzer(text, *, top_k)``.

    The worker remains responsible for optional extraction, lease ownership,
    persistence, and failure categorization. With the default adapter config,
    claim text is passed unchanged and no translation is introduced here.
    """
    if not callable(getattr(service, "run", None)):
        raise TypeError("service must provide async run")
    if multilingual_adapter is not None and not isinstance(multilingual_adapter, MultilingualClaimAdapter):
        raise TypeError("multilingual_adapter must be a MultilingualClaimAdapter or None")

    async def analyze(claim_text: str, *, top_k: int = 5) -> RagResult:
        input_data = ClaimInput(claim=claim_text)
        if multilingual_adapter is not None:
            input_data = await multilingual_adapter.aprepare_input(input_data)
        return await service.run(
            input_data,
            top_k=top_k,
            source_id=source_id,
            legal_domain=legal_domain,
            language=language,
        )

    return analyze


__all__ = ["AsyncRagService", "create_worker_rag_analyzer"]
