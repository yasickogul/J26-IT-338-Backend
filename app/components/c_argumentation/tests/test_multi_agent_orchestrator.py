from datetime import date
from uuid import uuid4

import pytest

from app.components.c_argumentation.pipeline.multi_agent_orchestrator import run_case_assessment
from app.components.c_argumentation.pipeline.retrieval_repository import RetrievedPassage


def _passage(case="CA/COL/0001/2020"):
    return RetrievedPassage(
        uuid4(), uuid4(), "A deed and possession passage.", 3, 4, case,
        date(2020, 1, 2), "https://example.test/judgment.pdf", 0.8,
    )


class FakeRetriever:
    def __init__(self, passages=None):
        self.passages = passages if passages is not None else [_passage(), _passage("CA/COL/0002/2021")]
        self.calls = []

    async def search_hybrid(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return self.passages

    @staticmethod
    def assess_evidence(passages):
        from app.components.c_argumentation.pipeline.evidence_quality import assess_evidence
        return assess_evidence(passages)


def _output(role):
    if role == "plaintiff":
        return '{"arguments":[{"claim":"The deed may support ownership.","supporting_principles":[],"evidence_source_ids":["S1"],"evidence_support":["[S1]"],"assumptions":[],"missing_evidence":["Original deed"],"limitations":[]}],"limitations":[],"unsupported_claim_warnings":[]}'
    if role == "defense":
        return '{"counterarguments":[{"challenged_claim":"The deed proves title.","counterargument":"Authenticity and title chain require proof.","evidence_source_ids":["S1"],"factual_disputes":[],"possible_defenses":[],"missing_evidence":["Title search"],"limitations":[]}],"limitations":[],"unsupported_claim_warnings":[]}'
    return '{"strengths":["The passage is relevant."],"weaknesses":["The record is limited."],"missing_evidence":["Original deed"],"unsupported_claim_warnings":[],"contradictory_claims":[],"verified_source_ids":["S1"],"evidence_sufficiency":"limited","assessment":"No verdict is possible from these passages.","disclaimer":"Informational legal analysis only; not professional legal advice or a court verdict."}'


class FakeLLM:
    def __init__(self, outputs=None):
        self.outputs = outputs or [_output("plaintiff"), _output("defense"), _output("auditor")]
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return type("Result", (), {"text": self.outputs[len(self.prompts) - 1]})()


@pytest.mark.asyncio
async def test_agents_run_sequentially_and_reuse_one_model():
    retriever, llm = FakeRetriever(), FakeLLM()
    result = await run_case_assessment("Ownership dispute", "Old deed and possession.", retriever=retriever, llm=llm)
    assert len(llm.prompts) == 3
    assert "PLAINTIFF OUTPUT" in llm.prompts[1]
    assert "PLAINTIFF:" in llm.prompts[2] and "DEFENSE:" in llm.prompts[2]
    assert len(retriever.calls) == 1
    assert result.citation_audits["plaintiff"].is_valid is True


@pytest.mark.asyncio
async def test_prompts_stay_bounded_and_empty_evidence_is_insufficient():
    long_passage = _passage()
    long_passage = RetrievedPassage(
        long_passage.chunk_id, long_passage.judgment_id, "x" * 10000,
        long_passage.start_page, long_passage.end_page, long_passage.case_number,
        long_passage.judgment_date, long_passage.pdf_url, long_passage.similarity_score,
    )
    llm = FakeLLM()
    result = await run_case_assessment("Question", "Facts", retriever=FakeRetriever([long_passage]), llm=llm)
    assert all(len(prompt) < 6500 for prompt in llm.prompts)

    empty = await run_case_assessment("Question", retriever=FakeRetriever([]), llm=FakeLLM())
    assert empty.evidence_sufficiency == "insufficient"


@pytest.mark.asyncio
async def test_malformed_model_output_returns_safe_limited_assessment():
    result = await run_case_assessment("Question", retriever=FakeRetriever(), llm=FakeLLM(["not json"]))
    assert result.execution_errors
    assert result.evidence_sufficiency == "insufficient"
    assert result.disclaimer.startswith("Informational legal analysis")


@pytest.mark.asyncio
async def test_empty_user_input_is_rejected():
    with pytest.raises(ValueError, match="problem"):
        await run_case_assessment("   ", retriever=FakeRetriever(), llm=FakeLLM())
