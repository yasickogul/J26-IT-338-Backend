from app.components.c_argumentation.pipeline.citation_verifier import verify_citations


def test_citation_verifier_accepts_known_source():
    result = verify_citations({"cites": ["src-1"]}, [{"id": "src-1"}])
    assert result["valid"] is True
