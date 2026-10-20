from app.components.c_argumentation.pipeline.source_retriever import retrieve_sources


def test_retrieve_sources_copies_records():
    source = {"id": "src-1"}
    result = retrieve_sources({}, [source])
    assert result == [source]
    assert result[0] is not source
