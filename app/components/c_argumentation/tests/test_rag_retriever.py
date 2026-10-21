from __future__ import annotations

from datetime import date
from uuid import uuid4

from pytest import approx

from app.components.c_argumentation.pipeline import rag_retriever
from app.components.c_argumentation.pipeline.retrieval_repository import (
    FULL_TEXT_RETRIEVAL_SQL,
    RETRIEVAL_SQL,
    RetrievedPassage,
    build_issue_aware_full_text_query,
    _fuse_rankings,
    build_full_text_query,
    detect_legal_issue,
    diversify_passages,
    rerank_by_legal_issue,
    search_judgment_chunks,
)


class FakeResult:
    def mappings(self):
        return self

    def all(self):
        return [{
            "chunk_id": uuid4(),
            "judgment_id": uuid4(),
            "chunk_text": "A court may consider a registered conveyance.",
            "start_page": 3,
            "end_page": 4,
            "case_number": "CA/COL/0001/2020",
            "judgment_date": date(2020, 1, 2),
            "pdf_url": "https://example.test/judgment.pdf",
            "similarity_score": 0.91,
        }]


class FakeSession:
    last_statement = None
    last_parameters = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, statement, parameters):
        type(self).last_statement = statement
        type(self).last_parameters = parameters
        return FakeResult()


def fake_session_factory():
    return FakeSession()


async def test_retrieval_returns_case_and_page_metadata():
    results = await search_judgment_chunks([0.0] * 384, session_factory=fake_session_factory)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, RetrievedPassage)
    assert result.case_number == "CA/COL/0001/2020"
    assert result.start_page == 3
    assert result.end_page == 4
    assert result.pdf_url.endswith("judgment.pdf")
    assert result.similarity_score == 0.91


def test_retrieval_sql_is_parameterized():
    sql = RETRIEVAL_SQL.text

    assert ":query_embedding" in sql
    assert ":candidate_limit" in sql
    assert "CAST(:query_embedding AS vector)" in sql
    assert ":query_text" in FULL_TEXT_RETRIEVAL_SQL.text
    assert "websearch_to_tsquery('english', :query_text)" in FULL_TEXT_RETRIEVAL_SQL.text
    assert build_full_text_query("registered deeds and property") == "registered OR deeds OR property"


def test_issue_aware_query_expansion_is_limited_to_known_issue_families():
    contract_query = build_issue_aware_full_text_query("What remedies follow a breach of contract?")
    evidence_query = build_issue_aware_full_text_query("Who bears the burden of proof?")

    assert detect_legal_issue("What remedies follow a breach of contract?") == "contract"
    assert detect_legal_issue("Who bears the burden of proof?") == "civil_evidence"
    assert "specific performance" in contract_query
    assert "admiss" in evidence_query
    assert build_issue_aware_full_text_query("What is the legal position?") == "legal OR position"


def test_issue_rerank_is_bounded_and_preserves_metadata():
    first = RetrievedPassage(uuid4(), uuid4(), "A contract was breached and damages were claimed.", 2, 3, "A", None, "https://example.test/a", 0.8, rrf_score=0.02)
    second = RetrievedPassage(uuid4(), uuid4(), "The deed was attested before a notary.", 4, 5, "B", None, "https://example.test/b", 0.9, rrf_score=0.021)

    ranked = rerank_by_legal_issue("What remedy follows breach of contract?", [first, second])

    assert ranked[0].case_number == "A"
    assert ranked[0].issue_score == 1.0
    assert ranked[0].rerank_score == approx(0.022)
    assert ranked[0].start_page == 2
    assert ranked[0].pdf_url.endswith("a")
    assert ranked[1].issue_score == 0.0


def test_rrf_fusion_combines_vector_and_full_text_ranks():
    first = {
        "chunk_id": "one",
        "judgment_id": uuid4(),
        "chunk_text": "one",
        "start_page": 1,
        "end_page": 1,
        "case_number": "A",
        "judgment_date": None,
        "pdf_url": None,
        "similarity_score": 0.9,
    }
    second = {
        "chunk_id": "two",
        "judgment_id": uuid4(),
        "chunk_text": "two",
        "start_page": 2,
        "end_page": 2,
        "case_number": "B",
        "judgment_date": None,
        "pdf_url": None,
        "similarity_score": 0.8,
    }
    text_first = {**first, "full_text_score": 1.0}
    fused = _fuse_rankings([first, second], [text_first], rrf_k=1)

    assert fused[0].chunk_id == "one"
    assert fused[0].case_number == "A"
    assert fused[0].start_page == 1
    assert fused[0].end_page == 1
    assert fused[0].vector_rank == 1
    assert fused[0].full_text_rank == 1
    assert fused[0].rrf_score == 1.0


def test_diversification_limits_one_judgment():
    judgment_id = uuid4()
    passages = [
        RetrievedPassage(uuid4(), judgment_id, "a", 1, 1, "A", None, None, 0.9),
        RetrievedPassage(uuid4(), judgment_id, "b", 2, 2, "A", None, None, 0.8),
        RetrievedPassage(uuid4(), uuid4(), "c", 3, 3, "B", None, None, 0.7),
    ]

    selected = diversify_passages(passages, top_k=2, max_per_judgment=1)

    assert [passage.case_number for passage in selected] == ["A", "B"]


async def test_empty_question_returns_no_results(monkeypatch):
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("empty questions must not be embedded")

    monkeypatch.setattr(rag_retriever, "embed_texts", fail_if_called)
    retriever = rag_retriever.C4RagRetriever()

    assert await retriever.search("   ") == []
    assert called is False


async def test_empty_embedding_returns_no_results():
    assert await search_judgment_chunks([], session_factory=fake_session_factory) == []


def test_empty_and_weak_evidence_do_not_support_legal_conclusion():
    from app.components.c_argumentation.pipeline.evidence_quality import assess_evidence

    empty = assess_evidence([])
    weak = assess_evidence([
        RetrievedPassage(uuid4(), uuid4(), "text", 1, 1, "A", None, None, 0.2)
    ])

    assert empty.label == "empty"
    assert weak.label == "weak"
    assert empty.supports_legal_conclusion is False
    assert weak.supports_legal_conclusion is False
