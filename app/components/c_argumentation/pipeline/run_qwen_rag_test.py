"""C4 RAG-grounded Qwen test using the existing hybrid retriever."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

from .local_qwen_adapter import LocalQwenAdapter
from .qwen_grounding import generate_grounded_answer
from .rag_retriever import C4RagRetriever


QUESTION = "What evidence is relevant when two parties dispute possession and ownership of land in Sri Lanka?"


async def main() -> None:
    started = time.perf_counter()
    retriever = C4RagRetriever()
    passages = await retriever.search_hybrid(QUESTION, top_k=2, candidate_limit=50, max_per_judgment=1)
    retrieval_seconds = time.perf_counter() - started
    adapter = LocalQwenAdapter(Path(__file__).resolve().parents[4] / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    started = time.perf_counter()
    answer, citations = generate_grounded_answer(adapter, QUESTION, passages, max_tokens=256)
    inference_seconds = time.perf_counter() - started
    print(f"RETRIEVAL_SECONDS={retrieval_seconds:.3f}")
    for index, passage in enumerate(passages, start=1):
        print(
            f"SOURCE_{index}=case={passage.case_number};pages={passage.start_page}-{passage.end_page};"
            f"similarity={passage.similarity_score};rrf={passage.rrf_score};"
            f"excerpt={passage.chunk_text[:220].replace(chr(10), ' ')}"
        )
    print(f"INFERENCE_SECONDS={inference_seconds:.3f}")
    print(f"CITATIONS_VALID={citations.is_valid}")
    print(f"VALID_SOURCE_NUMBERS={citations.valid_source_numbers}")
    print(f"INVALID_SOURCE_NUMBERS={citations.invalid_source_numbers}")
    print(f"INVALID_CASE_NUMBERS={citations.invalid_case_numbers}")
    print(f"ANSWER={answer.replace(chr(10), ' ')}")


if __name__ == "__main__":
    asyncio.run(main())
