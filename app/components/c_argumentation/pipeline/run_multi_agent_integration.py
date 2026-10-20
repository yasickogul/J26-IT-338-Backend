"""Exactly one manual C4 multi-agent integration run; not a production endpoint."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time

from .local_qwen_adapter import LocalQwenAdapter
from .multi_agent_orchestrator import run_case_assessment
from .rag_retriever import C4RagRetriever


SCENARIO = (
    "A person claims ownership of a piece of land based on an old deed and long-term possession. "
    "Another person disputes the ownership and claims the deed is invalid. What are the arguments "
    "supporting and opposing the first person's claim, and what evidence is missing?"
)


async def main() -> None:
    started = time.perf_counter()
    adapter = LocalQwenAdapter(
        Path(__file__).resolve().parents[4] / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
    )
    retriever = C4RagRetriever()
    assessment = await run_case_assessment(
        SCENARIO,
        SCENARIO,
        retriever=retriever,
        llm=adapter,
        top_k=3,
    )
    total_seconds = time.perf_counter() - started
    print(f"TOTAL_SECONDS={total_seconds:.3f}")
    print("RETRIEVED_SOURCES=")
    for source in assessment.verified_court_case_references:
        print(json.dumps({
            "source_id": source.source_id,
            "case_number": source.case_number,
            "start_page": source.start_page,
            "end_page": source.end_page,
            "similarity_score": source.similarity_score,
            "pdf_url": source.pdf_url,
            "excerpt": source.chunk_text[:240].replace(chr(10), " "),
        }, ensure_ascii=False))
    print("ASSESSMENT_JSON=")
    if hasattr(assessment, "model_dump_json"):
        print(assessment.model_dump_json(indent=2))
    else:
        print(assessment.json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
