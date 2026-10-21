from .evaluate_qwen_evidence_sets import DISPLAY_TOP_K, LEXICAL_POOL, RRF_K, SEMANTIC_POOL, _synthetic_unrelated_packet


def test_evaluation_settings_and_synthetic_packet_are_evaluation_only():
    assert (SEMANTIC_POOL, LEXICAL_POOL, RRF_K, DISPLAY_TOP_K) == (50, 50, 60, 5)
    packet = _synthetic_unrelated_packet("Employees in Sri Lanka are entitled to gratuity.")
    assert packet["retrieval_settings"]["synthetic"] is True
    assert packet["evidence"][0]["evidence_id"] == "chunk:synthetic-unrelated-1"
    assert packet["evidence"][0]["text"]
