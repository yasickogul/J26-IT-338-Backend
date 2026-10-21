
"""C4 adapter for the shared legal retrieval service."""

from collections.abc import Iterable, Mapping
from typing import Protocol

from app.shared.contracts.retrieval import RetrievedChunk


class LegalRetriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: list[str] | None = None,
        filters: dict | None = None,
        alpha: float = 0.5,
    ) -> list[RetrievedChunk]:
        ...


def retrieve_sources(
    case_facts: Mapping[str, object],
    sources: Iterable[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Preserve the existing source-record adapter."""
    del case_facts
    return [dict(source) for source in sources]


def build_legal_query(case_data: Mapping[str, object]) -> str:
    """Create a retrieval query from submitted case information."""

    fields = (
        "dispute_type",
        "facts",
        "user_position",
        "opposing_position",
    )

    parts = [
        str(case_data[field]).strip()
        for field in fields
        if case_data.get(field)
    ]

    return "\n".join(parts)


async def retrieve_legal_sources(
    case_data: Mapping[str, object],
    retriever: LegalRetriever,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Retrieve relevant legal chunks through the shared interface."""

    query = build_legal_query(case_data)

    if not query:
        raise ValueError("Case information is required for retrieval.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    return await retriever.search(
        query,
        top_k=top_k,
        source_types=["act", "judgment", "regulation"],
    )
