from app.components.c_argumentation.pipeline.agent_common import audit_text_citations
from app.components.c_argumentation.schemas_multi_agent import EvidenceBundle, EvidenceSource
from uuid import uuid4


def _bundle():
    return EvidenceBundle(
        sources=[EvidenceSource(source_id="S1", chunk_id=uuid4(), judgment_id=uuid4(), case_number="CA/COL/0001/2020", start_page=1, end_page=2, pdf_url=None, similarity_score=0.7, chunk_text="text")],
        evidence_label="weak",
        evidence_explanation="limited",
    )


def test_unsupported_source_and_case_are_rejected():
    audit = audit_text_citations("Use S1 but not S9 or CA/UNKNOWN/9.", _bundle())
    assert audit.valid_source_ids == ["S1"]
    assert audit.invalid_source_ids == ["S9"]
    assert audit.invalid_case_numbers == ["CA/UNKNOWN/9"]
    assert audit.is_valid is False


def test_known_case_reference_is_accepted():
    audit = audit_text_citations("See [S1] and CA/COL/0001/2020.", _bundle())
    assert audit.is_valid is True
