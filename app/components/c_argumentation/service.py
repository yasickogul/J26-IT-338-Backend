
from .schemas import CaseInput, CaseResponse
from .pipeline.case_intake import process_case
from .pipeline.multi_agent_orchestrator import run_case_assessment
from .schemas_multi_agent import MultiAgentAssessment


def submit_case(case: CaseInput) -> CaseResponse:
    return process_case(case)


async def assess_case(case: CaseInput) -> MultiAgentAssessment:
    """Run the C4 RAG and sequential agent assessment for the C4 API only."""

    problem = (
        f"Case title: {case.case_title.strip()}\n"
        f"Dispute type: {case.dispute_type.strip()}\n"
        f"User position: {case.user_position.strip()}\n"
        f"Opposing position: {(case.opposing_position or 'Not provided').strip()}"
    )
    return await run_case_assessment(problem, case.case_facts, documents=case.documents)
