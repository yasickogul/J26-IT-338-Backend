"""Exactly one real full C4 workflow validation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time

from .local_qwen_adapter import LocalQwenAdapter
from .multi_agent_orchestrator import run_case_assessment
from .rag_retriever import C4RagRetriever


QUESTION = (
    "A person claims ownership of land based on an old deed and long-term possession. "
    "Another person disputes the deed's validity. What arguments support and challenge "
    "the ownership claim, and what evidence is missing?"
)


class RecordingLLM:
    def __init__(self, adapter: LocalQwenAdapter) -> None:
        self.adapter = adapter
        self.records: list[dict[str, object]] = []

    def generate(self, prompt: str, **kwargs):
        started = time.perf_counter()
        result = self.adapter.generate(prompt, **kwargs)
        self.records.append({
            "seconds": round(time.perf_counter() - started, 3),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "prompt_chars": len(prompt),
        })
        return result


async def main() -> None:
    adapter = LocalQwenAdapter(
        Path(__file__).resolve().parents[4] / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
    )
    llm = RecordingLLM(adapter)
    started = time.perf_counter()
    assessment = await run_case_assessment(
        QUESTION,
        QUESTION,
        retriever=C4RagRetriever(),
        llm=llm,
        top_k=3,
    )
    print(f"TOTAL_SECONDS={time.perf_counter() - started:.3f}")
    print("SOURCES=")
    for source in assessment.verified_court_case_references:
        print(f"{source.source_id} {source.case_number} pages={source.start_page}-{source.end_page} similarity={source.similarity_score}")
    print(f"AGENT_RECORDS={llm.records}")
    print(f"FALLBACK={bool(assessment.execution_errors)}")
    print(f"EXECUTION_ERRORS={assessment.execution_errors}")
    print(f"EVIDENCE_SUFFICIENCY={assessment.evidence_sufficiency}")
    print("CITATION_AUDITS=")
    for role, audit in assessment.citation_audits.items():
        print(f"{role}={audit.model_dump() if hasattr(audit, 'model_dump') else audit.dict()}")
    print("PLAINTIFF=")
    print(assessment.plaintiff_arguments.model_dump_json() if hasattr(assessment.plaintiff_arguments, "model_dump_json") else assessment.plaintiff_arguments.json())
    print("DEFENSE=")
    print(assessment.defense_counterarguments.model_dump_json() if hasattr(assessment.defense_counterarguments, "model_dump_json") else assessment.defense_counterarguments.json())
    print("AUDITOR=")
    print(assessment.neutral_case_assessment.model_dump_json() if hasattr(assessment.neutral_case_assessment, "model_dump_json") else assessment.neutral_case_assessment.json())
    print("FULL_ASSESSMENT=")
    print(json.dumps({
        "strengths": assessment.strengths,
        "weaknesses": assessment.weaknesses,
        "missing_evidence": assessment.missing_evidence,
        "unsupported_claim_warnings": assessment.unsupported_claim_warnings,
        "disclaimer": assessment.disclaimer,
    }))


if __name__ == "__main__":
    asyncio.run(main())
