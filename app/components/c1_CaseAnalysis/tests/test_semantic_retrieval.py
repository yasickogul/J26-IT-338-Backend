"""Unit tests for C1 semantic retrieval and grounded analysis (no live Neon/LLM required)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.components.c1_CaseAnalysis.analysis.grounded import build_grounded_analysis
from app.components.c1_CaseAnalysis.dev_app import app
from app.components.c1_CaseAnalysis.retrieval.query_builder import build_retrieval_query
from app.components.c1_CaseAnalysis.retrieval.semantic import (
    SemanticRetrievalError,
    SemanticRetriever,
    embed_query,
    matches_from_raw_rows,
    validate_top_k,
)
from app.components.c1_CaseAnalysis.schemas import (
    CaseInformation,
    ExtractedLegalInfo,
    SimilarJudgmentMatch,
)
from app.shared.contracts.legal_info import StructuredLegalInfo


def _sample_extracted() -> ExtractedLegalInfo:
    return ExtractedLegalInfo(
        intake_id="00000000-0000-0000-0000-000000000001",
        status="done",
        case_information=CaseInformation(
            case_type="Employment",
            dispute_category="employment",
            title="Nimal Perera v. Ceylon Petroleum Corporation",
            summary=(
                "An employee was interdicted with half pay after an internal investigation "
                "and challenges the Managing Director's authority."
            ),
            remedy_requested="Writs of Certiorari and Mandamus",
        ),
        legal_issues=[],
        claims=["Ultra vires interdiction with half pay"],
        evidence_mentioned=["Internal investigation report — available"],
        structured_legal_info=StructuredLegalInfo(
            claims=["Ultra vires interdiction with half pay"],
            parties=["Nimal Perera", "Ceylon Petroleum Corporation"],
            evidence=["Internal investigation report — available"],
        ),
        model_version="stage2-rules-v1",
        disclaimer="This is informational and not legal advice.",
    )


def test_validate_top_k_bounds():
    assert validate_top_k(None) == 5
    assert validate_top_k(3) == 3
    with pytest.raises(ValueError):
        validate_top_k(0)
    with pytest.raises(ValueError):
        validate_top_k(99)


def test_build_retrieval_query_uses_only_supplied_facts():
    query = build_retrieval_query(
        description="Supplier delivered materials and was not paid after written demand.",
        legal_issue="Breach of contract involving non-payment",
        dispute_category="contract",
        evidence_factors=["Written agreement — available"],
    )
    assert "Breach of contract" in query
    assert "Written agreement" in query
    assert "CA/WRT" not in query


def test_build_retrieval_query_rejects_empty():
    with pytest.raises(ValueError, match="RETRIEVAL_QUERY_EMPTY"):
        build_retrieval_query()


@pytest.mark.asyncio
async def test_embed_query_dimension_and_finite_values():
    fake_vector = [0.01] * 384

    class FakeEmbedder:
        def embed_documents(self, texts):
            assert texts and texts[0]
            return [fake_vector]

    with patch(
        "app.components.c1_CaseAnalysis.retrieval.semantic.get_embedder",
        return_value=FakeEmbedder(),
    ):
        vector = await embed_query("A civil employment dispute involving interdiction.")
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)


@pytest.mark.asyncio
async def test_embed_query_rejects_bad_dimension():
    class FakeEmbedder:
        def embed_documents(self, texts):
            return [[0.1, 0.2]]

    with patch(
        "app.components.c1_CaseAnalysis.retrieval.semantic.get_embedder",
        return_value=FakeEmbedder(),
    ), pytest.raises(SemanticRetrievalError) as exc:
        await embed_query("Enough characters for a meaningful query here.")
    assert exc.value.code == "EMBEDDING_DIM_MISMATCH"


def test_distinct_judgment_grouping_from_mocked_rows():
    rows = [
        {
            "document_id": "doc-1",
            "title": "Case One",
            "citation": "CA/WRT/0001/2024",
            "court": "Court of Appeal",
            "decision_date": "2024-01-01",
            "source_url": "https://example.test/1",
            "source_type": "judgment",
            "case_type": "civil",
            "chunk_id": "chunk-a",
            "chunk_index": 0,
            "section": None,
            "chunk_text": "Supporting excerpt one.",
            "similarity": 0.81,
        },
        {
            "document_id": "doc-2",
            "title": "Case Two",
            "citation": "CA/WRT/0002/2024",
            "court": "Court of Appeal",
            "decision_date": "2024-02-01",
            "source_url": None,
            "source_type": "judgment",
            "case_type": "civil",
            "chunk_id": "chunk-b",
            "chunk_index": 3,
            "section": "ratio",
            "chunk_text": "Supporting excerpt two.",
            "similarity": 0.77,
        },
    ]
    matches = matches_from_raw_rows(rows)
    assert len(matches) == 2
    assert matches[0].citation == "CA/WRT/0001/2024"
    assert matches[0].source_url == "https://example.test/1"
    assert matches[0].score_label == "retrieval_similarity_cosine"
    assert "success" not in matches[0].score_label


@pytest.mark.asyncio
async def test_semantic_retriever_maps_db_rows():
    session = AsyncMock()
    result = MagicMock()
    result.mappings.return_value.all.return_value = [
        {
            "document_id": "08494366-0000-0000-0000-000000000001",
            "title": "Sample v. Corporation",
            "citation": "CA/WRT/0056/2025",
            "court": "Court of Appeal",
            "decision_date": None,
            "source_url": "https://example.test/judgment",
            "source_type": "judgment",
            "case_type": "civil",
            "chunk_id": "chunk-1",
            "chunk_index": 2,
            "section": None,
            "chunk_text": "Interdiction with half pay pending disciplinary proceedings.",
            "similarity": 0.73,
        }
    ]
    session.execute = AsyncMock(return_value=result)

    with patch(
        "app.components.c1_CaseAnalysis.retrieval.semantic.embed_query",
        new=AsyncMock(return_value=[0.0] * 384),
    ):
        matches = await SemanticRetriever().search(
            session,
            "Employee interdicted with half pay challenges authority.",
            top_k=5,
        )
    assert len(matches) == 1
    assert matches[0].citation == "CA/WRT/0056/2025"
    assert matches[0].retrieval_similarity == 0.73
    session.execute.assert_awaited()


@pytest.mark.asyncio
async def test_semantic_retriever_database_failure():
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=RuntimeError("db down"))
    with patch(
        "app.components.c1_CaseAnalysis.retrieval.semantic.embed_query",
        new=AsyncMock(return_value=[0.0] * 384),
    ), pytest.raises(SemanticRetrievalError) as exc:
        await SemanticRetriever().search(session, "A sufficiently long query for retrieval.")
    assert exc.value.code == "DATABASE_SEARCH_FAILED"


@pytest.mark.asyncio
async def test_grounded_analysis_without_llm():
    matches = [
        SimilarJudgmentMatch(
            document_id="doc-1",
            title="Sample Judgment",
            citation="CA/WRT/0056/2025",
            court="Court of Appeal",
            decision_date="2026-06-05",
            source_url=None,
            chunk_id="c1",
            chunk_index=1,
            supporting_excerpt="The Managing Director's authority to interdict was considered.",
            retrieval_similarity=0.8,
        )
    ]
    analysis = await build_grounded_analysis(
        extracted=_sample_extracted(),
        matches=matches,
        use_llm=False,
    )
    assert analysis.llm_used is False
    assert analysis.llm_status == "not_requested"
    assert analysis.similar_historical_cases
    assert "CA/WRT/0056/2025" in analysis.similar_historical_cases[0].title
    assert any("not a prediction" in lim.lower() or "semantic" in lim.lower() for lim in analysis.limitations)


@pytest.mark.asyncio
async def test_lm_studio_fallback_when_disabled():
    analysis = await build_grounded_analysis(
        extracted=_sample_extracted(),
        matches=[],
        use_llm=True,
    )
    # Default C1_LM_STUDIO_ENABLED=false → disabled path
    assert analysis.llm_used is False
    assert analysis.llm_status in {"disabled", "not_requested", "unavailable_no_model"} or analysis.llm_status.startswith(
        "unavailable"
    )


def test_similar_cases_endpoint_validation():
    client = TestClient(app)
    r = client.post("/api/v1/c1/similar-cases", json={"query": "too short", "top_k": 5})
    assert r.status_code == 422


def test_intake_similar_cases_missing_id():
    client = TestClient(app)
    r = client.post("/api/v1/c1/intake/not-a-uuid/similar-cases", json={"top_k": 3})
    assert r.status_code in {404, 422}


def test_c1_router_imports_and_registers_similar_cases_routes():
    """Regression: schemas NameError previously prevented the whole C1 router from loading."""
    from app.components.c1_CaseAnalysis import router as c1_router
    from app.components.c1_CaseAnalysis.schemas import (
        IntakeSimilarCasesResponse,
        SimilarCasesResponse,
    )

    paths = {getattr(route, "path", None) for route in c1_router.router.routes}
    assert "/similar-cases" in paths
    assert "/intake/{intake_id}/similar-cases" in paths
    # Instantiating response models must not raise NameError for score defaults.
    SimilarCasesResponse(query="x" * 20, retrieved_count=0, cases=[])
    IntakeSimilarCasesResponse(intake_id="00000000-0000-0000-0000-000000000001", query="x" * 20, retrieved_count=0)
