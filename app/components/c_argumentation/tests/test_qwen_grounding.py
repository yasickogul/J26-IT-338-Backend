from datetime import date
from uuid import uuid4

from app.components.c_argumentation.pipeline.qwen_grounding import (
    build_grounded_prompt,
    generate_grounded_answer,
    validate_citations,
)
from app.components.c_argumentation.pipeline.retrieval_repository import RetrievedPassage


def passage(case="CA/COL/0001/2020"):
    return RetrievedPassage(uuid4(), uuid4(), "Possession evidence text.", 3, 4, case, date(2020, 1, 2), "https://example.test/a.pdf", 0.8)


def test_prompt_preserves_case_and_pages_and_grounding_rules():
    prompt = build_grounded_prompt("What evidence matters?", [passage()])
    assert "CA/COL/0001/2020" in prompt
    assert "Pages: 3-4" in prompt
    assert "using only the supplied judgment passages" in prompt


def test_citation_validation_rejects_unknown_sources_and_cases():
    result = validate_citations("See [Source 1] and [Source 9], CA/UNKNOWN/1.", [passage()])
    assert result.valid_source_numbers == (1,)
    assert result.invalid_source_numbers == (9,)
    assert result.invalid_case_numbers == ("CA/UNKNOWN/1",)
    assert result.is_valid is False


def test_mocked_generator_returns_structured_grounded_answer():
    class FakeGenerator:
        def generate(self, prompt, **kwargs):
            assert "Evidence:" in prompt
            return type("Result", (), {"text": "Possession is discussed in [Source 1]."})()

    response, citations = generate_grounded_answer(FakeGenerator(), "What evidence matters?", [passage()])
    assert response.startswith("Possession")
    assert citations.is_valid is True


def test_empty_evidence_prompt_is_explicit():
    assert "evidence is insufficient" in build_grounded_prompt("Question?", [])
