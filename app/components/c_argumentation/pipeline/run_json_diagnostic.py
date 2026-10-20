"""Minimal real-model JSON-mode diagnostic for the C4 Qwen adapter."""

from __future__ import annotations

import json
from pathlib import Path
import time

from pydantic import BaseModel, ConfigDict, Field

from .local_qwen_adapter import LocalQwenAdapter


class MinimalJson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    argument: str = Field(min_length=1)
    source_ids: list[str]


def main() -> None:
    path = Path(__file__).resolve().parents[4] / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    adapter = LocalQwenAdapter(path, n_ctx=2048, n_threads=4)
    started = time.perf_counter()
    model = adapter.model
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = adapter.generate(
        'Return exactly one JSON object with this shape: {"argument":"A short legal argument","source_ids":["S1"]}. '
        'Use only source ID S1. Do not include Markdown, explanations, or any other keys.',
        max_tokens=96,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    inference_seconds = time.perf_counter() - started
    raw = result.text
    parsed = None
    error = None
    try:
        parsed = MinimalJson.model_validate(json.loads(raw))
    except Exception as exc:
        error = str(exc)
    metadata = getattr(model, "metadata", {})
    print(f"MODEL_PATH={path}")
    print(f"MODEL_BYTES={path.stat().st_size}")
    print(f"LOAD_SECONDS={load_seconds:.3f}")
    print(f"INFERENCE_SECONDS={inference_seconds:.3f}")
    print(f"PROMPT_TOKENS={result.prompt_tokens}")
    print(f"COMPLETION_TOKENS={result.completion_tokens}")
    print(f"CHAT_TEMPLATE={metadata.get('tokenizer.chat_template', 'unavailable')}")
    print(f"RAW_JSON={raw}")
    print(f"HAS_MARKDOWN_FENCE={'```' in raw}")
    print(f"VALID_JSON={parsed is not None}")
    print(f"SCHEMA_ERROR={error}")
    print(f"SOURCE_IDS={parsed.source_ids if parsed else None}")


if __name__ == "__main__":
    main()
