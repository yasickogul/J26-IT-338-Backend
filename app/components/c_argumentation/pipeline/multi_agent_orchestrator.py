"""C4-only sequential Plaintiff/Defense/Auditor orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agents.auditor_agent import NeutralAuditorAgent
from ..agents.defense_agent import DefenseAgent
from ..agents.plaintiff_agent import PlaintiffAgent
from ..schemas_multi_agent import (
    AuditorOutput,
    CitationAudit,
    DefenseOutput,
    EvidenceBundle,
    MultiAgentAssessment,
    PlaintiffOutput,
)
from ..schemas_document import DocumentEvidenceInput
from .agent_common import audit_structured_citations, make_evidence_bundle
from .evidence_quality import assess_evidence
from .local_qwen_adapter import LocalQwenAdapter
from .rag_retriever import C4RagRetriever


DISCLAIMER = "Informational legal analysis only; not professional legal advice or a court verdict."


@dataclass(frozen=True, slots=True)
class AgentExecution:
    plaintiff: PlaintiffOutput
    defense: DefenseOutput
    auditor: AuditorOutput
    citations: dict[str, CitationAudit]


def _safe_empty_outputs(error: str) -> AgentExecution:
    plaintiff = PlaintiffOutput(limitations=[error], unsupported_claim_warnings=[error])
    defense = DefenseOutput(limitations=[error], unsupported_claim_warnings=[error])
    auditor = AuditorOutput(
        missing_evidence=[error],
        unsupported_claim_warnings=[error],
        evidence_sufficiency="insufficient",
        assessment="The model output could not be safely parsed, so no substantive assessment is provided.",
    )
    return AgentExecution(plaintiff, defense, auditor, {})


async def run_case_assessment(
    problem: str,
    facts: str | None = None,
    *,
    retriever: C4RagRetriever | None = None,
    llm: Any | None = None,
    top_k: int = 3,
    documents: list[DocumentEvidenceInput] | None = None,
) -> MultiAgentAssessment:
    if not problem.strip():
        raise ValueError("problem must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    stated_facts = facts.strip() if facts and facts.strip() else problem.strip()
    retriever = retriever or C4RagRetriever()
    llm = llm or LocalQwenAdapter()
    passages = await retriever.search_hybrid(problem, top_k=top_k, max_per_judgment=1)
    quality = retriever.assess_evidence(passages)
    bundle = make_evidence_bundle(passages, quality, documents)

    execution_errors: list[str] = []
    try:
        plaintiff = PlaintiffAgent(llm).run(problem, stated_facts, bundle)
    except Exception as exc:
        execution_errors.append(f"plaintiff: {exc}")
        plaintiff = PlaintiffOutput(limitations=[str(exc)], unsupported_claim_warnings=[str(exc)])
    try:
        defense = DefenseAgent(llm).run(problem, stated_facts, bundle, plaintiff)
    except Exception as exc:
        execution_errors.append(f"defense: {exc}")
        defense = DefenseOutput(limitations=[str(exc)], unsupported_claim_warnings=[str(exc)])
    try:
        auditor = NeutralAuditorAgent(llm).run(problem, stated_facts, bundle, plaintiff, defense)
    except Exception as exc:
        execution_errors.append(f"auditor: {exc}")
        auditor = AuditorOutput(
            missing_evidence=[str(exc)],
            unsupported_claim_warnings=[str(exc)],
            evidence_sufficiency="insufficient",
            assessment="The model output could not be safely parsed, so no substantive assessment is provided.",
        )
    execution = AgentExecution(
        plaintiff,
        defense,
        auditor,
        {
            "plaintiff": audit_structured_citations(plaintiff, bundle),
            "defense": audit_structured_citations(defense, bundle),
            "auditor": audit_structured_citations(auditor, bundle),
        },
    )

    all_warnings = (
        execution.plaintiff.unsupported_claim_warnings
        + execution.defense.unsupported_claim_warnings
        + execution.auditor.unsupported_claim_warnings
    )
    for role, audit in execution.citations.items():
        if not audit.is_valid or audit.invalid_page_references:
            all_warnings.append(
                f"{role} output contained unsupported references: "
                f"source IDs={audit.invalid_source_ids}, cases={audit.invalid_case_numbers}, "
                f"pages={audit.invalid_page_references}."
            )
    return MultiAgentAssessment(
        case_summary=problem.strip(),
        stated_facts=stated_facts,
        plaintiff_arguments=execution.plaintiff,
        defense_counterarguments=execution.defense,
        strengths=execution.auditor.strengths,
        weaknesses=execution.auditor.weaknesses,
        missing_evidence=execution.auditor.missing_evidence,
        verified_court_case_references=bundle.sources,
        uploaded_document_references=bundle.document_sources,
        document_context_warnings=bundle.context_warnings,
        unsupported_claim_warnings=all_warnings,
        evidence_sufficiency=execution.auditor.evidence_sufficiency if passages else "insufficient",
        neutral_case_assessment=execution.auditor,
        citation_audits=execution.citations,
        execution_errors=execution_errors,
        disclaimer=DISCLAIMER,
    )
