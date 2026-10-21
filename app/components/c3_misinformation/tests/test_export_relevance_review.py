.\.venv\Scripts\python.exe -m app.components.c3_misinformation.tests.export_relevance_reviewfrom types import SimpleNamespace

from .export_relevance_review import (
    LEXICAL_CANDIDATE_POOL,
    RRF_K,
    SEMANTIC_CANDIDATE_POOL,
    _merge_ranked_passages,
)


def _row(chunk_id: str):
    return SimpleNamespace(chunk_id=chunk_id, metadata={"source_id": "SYN"}, text=f"passage {chunk_id}")


def test_export_preserves_supplied_ranked_results_and_settings():
    semantic = [_row("semantic-top")]
    fused = [(0.9, _row("EMP-010-0010"), 7, 1), (0.8, _row("semantic-top"), 1, 3)]
    passages = _merge_ranked_passages(semantic, fused)
    by_id = {item["chunk_id"]: item for item in passages}
    assert by_id["EMP-010-0010"]["baseline_rrf_rank"] == 1
    assert by_id["EMP-010-0010"]["semantic_rank"] == 7
    assert by_id["semantic-top"]["semantic_rank"] == 1
    assert by_id["semantic-top"]["baseline_rrf_rank"] == 2
    assert (SEMANTIC_CANDIDATE_POOL, LEXICAL_CANDIDATE_POOL, RRF_K) == (50, 50, 60)
