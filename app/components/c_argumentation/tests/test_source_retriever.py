
import asyncio

from app.shared.contracts.retrieval import (
    RetrievedChunk,
    SourceCitation,
)

from app.components.c_argumentation.pipeline.source_retriever import (
    retrieve_legal_sources,
)


class MockLegalRetriever:
    """Fake legal retriever for testing C4 without a database."""

    async def search(
        self,
        query,
        *,
        top_k=10,
        source_types=None,
        filters=None,
        alpha=0.5,
    ):
        assert "property_ownership" in query
        assert top_k == 5
        assert source_types == ["act", "judgment", "regulation"]

        return [
            RetrievedChunk(
                chunk_id="demo-chunk-001",
                text="Illustrative legal text for testing only.",
                score=0.95,
                citation=SourceCitation(
                    source_id="demo-source-001",
                    title="Mock Legal Source",
                    source_type="act",
                ),
            )
        ]


def test_retrieve_legal_sources():
    """Check that C4 can use the shared retrieval interface."""

    case_data = {
        "dispute_type": "property_ownership",
        "facts": "A buyer purchased land in Kandy.",
        "user_position": "The buyer claims ownership.",
        "opposing_position": "Another party disputes the sale.",
    }

    results = asyncio.run(
        retrieve_legal_sources(
            case_data,
            retriever=MockLegalRetriever(),
            top_k=5,
        )
    )

    assert len(results) == 1
    assert results[0].chunk_id == "demo-chunk-001"
    assert results[0].citation.source_type == "act"
