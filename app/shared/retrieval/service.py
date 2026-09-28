from app.shared.contracts.retrieval import RetrievedChunk


class HybridRetriever:
    """Keyword (tsvector) + vector (pgvector) search merged with RRF.
    TODO (lead): implement. Components depend on this signature only."""

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: list[str] | None = None,
        filters: dict | None = None,
        alpha: float = 0.5,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError
