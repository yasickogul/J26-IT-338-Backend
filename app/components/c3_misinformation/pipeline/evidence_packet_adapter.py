"""Async evidence orchestration over required, caller-supplied dependencies."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .context_expansion import ContextExpansion
from .retrieval_contract import RetrievalResult
from .evidence_packet import _validate_query, assemble_evidence_packet


class AsyncEvidenceRetrieval(Protocol):
    async def __call__(
        self, query: str, *, top_k: int, source_id: str | None,
        legal_domain: str | None, language: str | None,
    ) -> list[RetrievalResult]:
        """Apply the C3 retrieval contract, including validation and ordering.

        Bind retrieval_adapter.retrieve's embedder and search dependencies
        before injection. Preserve LaBSE model filtering, metadata filters,
        distance/similarity semantics, complete text, and original chunk IDs.
        """


class AsyncContextExpansion(Protocol):
    async def __call__(
        self, ranked_results: Sequence[RetrievalResult], *,
        max_depth: int, max_passages: int,
    ) -> ContextExpansion:
        """Apply the C3 expansion contract, including bounds and same-source lookup.

        Bind context_expansion_adapter.expand_parent_context's parent_lookup
        before injection. Preserve all parent metadata and resolution notices.
        """


class AsyncOriginalChunkLookup(Protocol):
    async def __call__(self, chunk_id: str) -> str:
        """Return the exact original ID for exactly one matching parent chunk.

        Preserve source scalar_one semantics: missing or multiple rows raise,
        never fall back to chunk_id or omit provenance. This is a logical
        lookup contract, not an assumption about shared corpus columns.
        """


async def build_evidence_packet(
    query: str,
    top_k: int = 5,
    *,
    retrieve: AsyncEvidenceRetrieval,
    expand_parent_context: AsyncContextExpansion,
    original_chunk_lookup: AsyncOriginalChunkLookup,
    source_id: str | None = None,
    legal_domain: str | None = None,
    language: str | None = None,
    max_parent_depth: int = 1,
    max_parent_passages: int = 10,
) -> dict[str, Any]:
    """Retrieve, expand, resolve provenance, then reuse the original assembler.

    Dependencies must be nonblocking async callables. Callers own any sessions
    and read-only transaction scope, including consistency across all lookups;
    this function never begins, commits, rolls back, or closes a transaction.
    Errors and cancellation propagate. Validation order and untrimmed settings
    match CompNet: query, retrieval, expansion, provenance, assembly.
    """
    _validate_query(query)
    ranked = await retrieve(
        query, top_k=top_k, source_id=source_id,
        legal_domain=legal_domain, language=language,
    )
    expansion = await expand_parent_context(
        ranked, max_depth=max_parent_depth, max_passages=max_parent_passages,
    )
    originals = {}
    for cid in dict.fromkeys(p.parent_chunk_id for p in expansion.parents):
        originals[cid] = await original_chunk_lookup(cid)
    return assemble_evidence_packet(
        query, ranked, expansion, original_chunk_ids=originals,
        retrieval_settings={
            "top_k": top_k, "source_id": source_id,
            "legal_domain": legal_domain, "language": language,
            "max_parent_depth": max_parent_depth,
            "max_parent_passages": max_parent_passages,
        },
    )
