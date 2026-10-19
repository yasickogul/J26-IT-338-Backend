"""Unconnected original-ID SELECT for the original CompNet corpus contract."""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .postgres_vector_search import require_corpus_contract, corpus_table


async def lookup_original_chunk_id(
    chunk_id: str, *, session: AsyncSession, corpus_contract: str,
) -> str:
    """Bind session/contract for AsyncOriginalChunkLookup.

    Preserve scalar_one missing/ambiguous-row errors, with no fabricated ID.
    Source matching is performed by parent lookup and packet assembly. This
    interface has no source_id argument; globally unique textual chunk IDs and
    a consistent caller-owned read-only corpus view are therefore required.
    This reads provenance for the selected chunk, not its referenced original
    row; an original chunk need not itself be a separately stored passage.
    """
    require_corpus_contract(corpus_contract)
    result = await session.execute(text(
        f"SELECT original_chunk_id FROM {corpus_table(corpus_contract)} WHERE chunk_id = :chunk_id"
    ), {"chunk_id": chunk_id})
    original = result.scalar_one()
    if not isinstance(original, str) or not original.strip():
        raise ValueError("Invalid corpus original_chunk_id")
    return original
