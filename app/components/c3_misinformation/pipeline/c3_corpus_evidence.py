"""C3-owned legal-corpus wiring for production claim verification."""

from functools import partial

from sqlalchemy.ext.asyncio import AsyncSession

from .context_expansion_adapter import expand_parent_context
from .evidence_packet_adapter import build_evidence_packet
from .postgres_parent_lookup import lookup_parents
from .postgres_provenance_lookup import lookup_original_chunk_id
from .postgres_vector_search import C3_CORPUS_CONTRACT, search
from .retrieval_adapter import AsyncQueryEmbedder, retrieve


async def build_c3_evidence_packet(
    query: str,
    *,
    session: AsyncSession,
    embedder: AsyncQueryEmbedder,
    query_model: str,
    top_k: int = 5,
    source_id: str | None = None,
    legal_domain: str | None = None,
    language: str | None = None,
    max_parent_depth: int = 1,
    max_parent_passages: int = 10,
):
    """Build evidence exclusively from ``c3.legal_chunks``.

    The caller owns one read-only transaction and session. There is no shared-
    corpus or public-corpus fallback.
    """
    bound = {"session": session, "corpus_contract": C3_CORPUS_CONTRACT}
    return await build_evidence_packet(
        query,
        top_k=top_k,
        retrieve=partial(
            retrieve,
            embedder=embedder,
            search=partial(search, query_model=query_model, **bound),
        ),
        expand_parent_context=partial(
            expand_parent_context,
            parent_lookup=partial(lookup_parents, **bound),
        ),
        original_chunk_lookup=partial(lookup_original_chunk_id, **bound),
        source_id=source_id,
        legal_domain=legal_domain,
        language=language,
        max_parent_depth=max_parent_depth,
        max_parent_passages=max_parent_passages,
    )


__all__ = ["build_c3_evidence_packet"]
