"""Explicit shared-corpus wiring; no engine, feature enablement or worker startup."""
from functools import partial
from sqlalchemy.ext.asyncio import AsyncSession
from .postgres_vector_search import search, SHARED_CORPUS_CONTRACT
from .postgres_parent_lookup import lookup_parents
from .postgres_provenance_lookup import lookup_original_chunk_id
from .retrieval_adapter import retrieve, AsyncQueryEmbedder
from .context_expansion_adapter import expand_parent_context
from .evidence_packet_adapter import build_evidence_packet


async def build_shared_evidence_packet(
    query: str, *, session: AsyncSession, embedder: AsyncQueryEmbedder,
    query_model: str, top_k: int = 5, source_id: str | None = None,
    legal_domain: str | None = None, language: str | None = None,
    max_parent_depth: int = 1, max_parent_passages: int = 10,
):
    """Caller supplies a read-only transaction and trusted async LaBSE encoder.

    Use one consistent corpus view throughout. This does not configure database
    isolation or close/commit/rollback the caller's session.
    """
    bound = dict(session=session, corpus_contract=SHARED_CORPUS_CONTRACT)
    return await build_evidence_packet(
        query, top_k=top_k,
        retrieve=partial(retrieve, embedder=embedder,
                         search=partial(search, query_model=query_model, **bound)),
        expand_parent_context=partial(expand_parent_context,
                                     parent_lookup=partial(lookup_parents, **bound)),
        original_chunk_lookup=partial(lookup_original_chunk_id, **bound),
        source_id=source_id, legal_domain=legal_domain, language=language,
        max_parent_depth=max_parent_depth, max_parent_passages=max_parent_passages,
    )
