"""Controlled real-model validation for the three C4 agents."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

from ..agents.auditor_agent import NeutralAuditorAgent
from ..agents.defense_agent import DefenseAgent
from ..agents.plaintiff_agent import PlaintiffAgent
from ..schemas_multi_agent import AuditorOutput, DefenseOutput, PlaintiffOutput
from .agent_common import make_evidence_bundle
from .evidence_quality import assess_evidence
from .local_qwen_adapter import LocalQwenAdapter
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
        try:
            result = self.adapter.generate(prompt, **kwargs)
        except Exception as exc:
            self.records.append({"seconds": time.perf_counter() - started, "error": str(exc), "prompt_chars": len(prompt), "raw": getattr(locals().get("result"), "text", None)})
            raise
        self.records.append({
            "seconds": time.perf_counter() - started,
            "prompt_chars": len(prompt),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "raw_chars": len(result.text),
            "raw": result.text,
        })
        return result


def _report(name: str, started: float, result: object | None, record: dict[str, object] | None, error: str | None) -> None:
    print(f"AGENT={name}")
    print(f"VALID={result is not None}")
    print(f"FALLBACK={error is not None}")
    print(f"SECONDS={time.perf_counter() - started:.3f}")
    print(f"RECORD={record}")
    if record and record.get("raw"):
        print(f"RAW_PREFIX={str(record['raw'])[:500]}")
        print(f"RAW_SUFFIX={str(record['raw'])[-200:]}")
    print(f"ERROR={error}")


async def main() -> None:
    retriever = C4RagRetriever()
    passages = await retriever.search_hybrid(QUESTION, top_k=3, max_per_judgment=1)
    bundle = make_evidence_bundle(passages, assess_evidence(passages))
    print("SOURCES=")
    for source in bundle.sources:
        print(f"{source.source_id} {source.case_number} pages={source.start_page}-{source.end_page}")

    adapter = LocalQwenAdapter(
        Path(__file__).resolve().parents[4] / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
    )
    llm = RecordingLLM(adapter)
    plaintiff = None
    defense = None
    for name, call in (
        ("plaintiff", lambda: PlaintiffAgent(llm).run(QUESTION, QUESTION, bundle)),
        ("defense", lambda: DefenseAgent(llm).run(QUESTION, QUESTION, bundle, plaintiff)),
        ("auditor", lambda: NeutralAuditorAgent(llm).run(QUESTION, QUESTION, bundle, plaintiff, defense)),
    ):
        started = time.perf_counter()
        error = None
        result = None
        try:
            result = call()
        except Exception as exc:
            error = str(exc)
        _report(name, started, result, llm.records[-1] if llm.records else None, error)
        if name == "plaintiff":
            plaintiff = result if isinstance(result, PlaintiffOutput) else None
        elif name == "defense":
            defense = result if isinstance(result, DefenseOutput) else None
        elif name == "auditor" and isinstance(result, AuditorOutput):
            print(f"AUDITOR_SUFFICIENCY={result.evidence_sufficiency}")
        if error:
            break


if __name__ == "__main__":
    asyncio.run(main())
