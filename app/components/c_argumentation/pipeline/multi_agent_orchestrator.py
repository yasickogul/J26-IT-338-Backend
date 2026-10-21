
"""C4-only sequential Plaintiff/Defense/Auditor orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
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


DISCLAIMER = (
    "Informational legal analysis only; "
    "not professional legal advice or a court verdict."
)


@dataclass(frozen=True, slots=True)
class AgentExecution:
    plaintiff: PlaintiffOutput
    defense: DefenseOutput
    auditor: AuditorOutput
    citations: dict[str, CitationAudit]


def _safe_empty_outputs(error: str) -> AgentExecution:
    plaintiff = PlaintiffOutput(
        limitations=[error],
        unsupported_claim_warnings=[error],
    )
    defense = DefenseOutput(
        limitations=[error],
        unsupported_claim_warnings=[error],
    )
    auditor = AuditorOutput(
        missing_evidence=[error],
        unsupported_claim_warnings=[error],
        evidence_sufficiency="insufficient",
        assessment=(
            "The model output could not be safely parsed, "
            "so no substantive assessment is provided."
        ),
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

    total_start = perf_counter()

    if not problem.strip():
        raise ValueError("problem must not be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    stated_facts = (
        facts.strip()
        if facts and facts.strip()
        else problem.strip()
    )

    retriever = retriever or C4RagRetriever()
    llm = llm or LocalQwenAdapter()

    # --------------------------------------------------
    # 1. Hybrid RAG retrieval
    # --------------------------------------------------

    rag_start = perf_counter()

    try:
        passages = await retriever.search_hybrid(
            problem,
            top_k=top_k,
            max_per_judgment=1,
        )
    finally:
        print(
            f"[C4 PERFORMANCE] Hybrid RAG: "
            f"{perf_counter() - rag_start:.2f} seconds",
            flush=True,
        )

    quality = retriever.assess_evidence(passages)
    bundle = make_evidence_bundle(
        passages,
        quality,
        documents,
    )

    execution_errors: list[str] = []

    # --------------------------------------------------
    # 2. Plaintiff Agent
    # --------------------------------------------------

    plaintiff_start = perf_counter()

    try:
        plaintiff = PlaintiffAgent(llm).run(
            problem,
            stated_facts,
            bundle,
        )

    except Exception as exc:
        execution_errors.append(f"plaintiff: {exc}")

        plaintiff = PlaintiffOutput(
            limitations=[str(exc)],
            unsupported_claim_warnings=[str(exc)],
        )

    finally:
        print(
            f"[C4 PERFORMANCE] Plaintiff Agent: "
            f"{perf_counter() - plaintiff_start:.2f} seconds",
            flush=True,
        )

    # --------------------------------------------------
    # 3. Defense Agent
    # --------------------------------------------------

    defense_start = perf_counter()

    try:
        defense = DefenseAgent(llm).run(
            problem,
            stated_facts,
            bundle,
            plaintiff,
        )

    except Exception as exc:
        execution_errors.append(f"defense: {exc}")

        defense = DefenseOutput(
            limitations=[str(exc)],
            unsupported_claim_warnings=[str(exc)],
        )

    finally:
        print(
            f"[C4 PERFORMANCE] Defense Agent: "
            f"{perf_counter() - defense_start:.2f} seconds",
            flush=True,
        )

    # --------------------------------------------------
    # 4. Neutral Auditor Agent
    # --------------------------------------------------

    auditor_start = perf_counter()

    try:
        auditor = NeutralAuditorAgent(llm).run(
            problem,
            stated_facts,
            bundle,
            plaintiff,
            defense,
        )

    except Exception as exc:
        execution_errors.append(f"auditor: {exc}")

        auditor = AuditorOutput(
            missing_evidence=[str(exc)],
            unsupported_claim_warnings=[str(exc)],
            evidence_sufficiency="insufficient",
            assessment=(
                "The model output could not be safely parsed, "
                "so no substantive assessment is provided."
            ),
        )

    finally:
        print(
            f"[C4 PERFORMANCE] Neutral Auditor: "
            f"{perf_counter() - auditor_start:.2f} seconds",
            flush=True,
        )

    # --------------------------------------------------
    # 5. Citation validation
    # --------------------------------------------------

    citation_start = perf_counter()

    try:
        execution = AgentExecution(
            plaintiff,
            defense,
            auditor,
            {
                "plaintiff": audit_structured_citations(
                    plaintiff, bundle
                ),
                "defense": audit_structured_citations(
                    defense, bundle
                ),
                "auditor": audit_structured_citations(
                    auditor, bundle
                ),
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
                    f"source IDs={audit.invalid_source_ids}, "
                    f"cases={audit.invalid_case_numbers}, "
                    f"pages={audit.invalid_page_references}."
                )

        result = MultiAgentAssessment(
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
            evidence_sufficiency=(
                execution.auditor.evidence_sufficiency
                if passages
                else "insufficient"
            ),
            neutral_case_assessment=execution.auditor,
            citation_audits=execution.citations,
            execution_errors=execution_errors,
            disclaimer=DISCLAIMER,
        )

    finally:
        print(
            f"[C4 PERFORMANCE] Citation Validation + Response: "
            f"{perf_counter() - citation_start:.2f} seconds",
            flush=True,
        )

        print(
            f"[C4 PERFORMANCE] TOTAL ASSESSMENT: "
            f"{perf_counter() - total_start:.2f} seconds",
            flush=True,
        )

    return result
