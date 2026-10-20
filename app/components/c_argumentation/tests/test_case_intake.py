from app.components.c_argumentation.pipeline.case_intake import parse_case_facts


def test_parse_case_facts_from_text():
    assert parse_case_facts("  facts  ") == {"facts": "facts"}
