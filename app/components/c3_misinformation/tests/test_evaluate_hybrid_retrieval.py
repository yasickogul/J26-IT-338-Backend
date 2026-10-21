from types import SimpleNamespace

from .evaluate_hybrid_retrieval import _build_lexical_query, _lexical_vector_sql, _page_label, _rrf


def test_lexical_query_uses_informative_terms_as_or_expression():
    query = _build_lexical_query("Employees in Sri Lanka are entitled to gratuity")
    assert query == "employees | entitled | gratuity | lanka | sri"
    assert " in " not in query


def test_lexical_query_is_empty_for_stopwords_only():
    assert _build_lexical_query("the and in of") == ""


def test_rrf_merges_shared_and_one_sided_candidates_with_component_ranks():
    semantic = [SimpleNamespace(chunk_id="shared"), SimpleNamespace(chunk_id="semantic-only")]
    lexical = [SimpleNamespace(chunk_id="shared", lexical_rank=1), SimpleNamespace(chunk_id="lexical-only", lexical_rank=2)]
    merged = _rrf(semantic, lexical, 60)
    by_id = {item.chunk_id: (score, srank, lrank) for score, item, srank, lrank in merged}
    assert by_id["shared"][1:] == (1, 1)
    assert by_id["semantic-only"][1:] == (2, None)
    assert by_id["lexical-only"][1:] == (None, 2)
    assert by_id["shared"][0] > by_id["semantic-only"][0]


def test_page_label_reads_metadata_and_section_fallback():
    assert _page_label({"page_number": 5}) == "5"
    assert _page_label({"section_label": "PDF page 6 > Part 1"}) == "6"
    assert _page_label({"section_label": "page unavailable"}) is None


def test_lexical_representations_make_duplication_and_metadata_weight_explicit():
    baseline = _lexical_vector_sql("baseline")
    normalized = _lexical_vector_sql("text_once_normalized")
    assert "embedding_text" in baseline and "coalesce(text" in baseline
    assert "embedding_text" not in normalized
    assert "setweight" in normalized and "section_label" in normalized
