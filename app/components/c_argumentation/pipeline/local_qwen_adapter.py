"""C4-local CPU adapter for the Qwen3 GGUF model.

This module intentionally has no dependency on the shared application LLM
configuration.  The model is discovered from an explicit path, an environment
variable, or the repository's local ``models`` directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any


MODEL_FILENAME = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
DEFAULT_N_CTX = 2048
DEFAULT_N_GPU_LAYERS = 0
DEFAULT_N_THREADS = 8


class QwenAdapterError(RuntimeError):
    """A model discovery, loading, or inference error."""


@dataclass(frozen=True, slots=True)
class QwenGeneration:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve and validate the local GGUF without embedding an absolute path."""

    if model_path:
        candidate = Path(model_path).expanduser().resolve()
        if candidate.is_file() and candidate.suffix.lower() == ".gguf":
            return candidate
        raise QwenAdapterError(f"GGUF not found at model path: {candidate}")

    candidates: list[Path] = []
    env_path = os.getenv("C_ARGUMENTATION_QWEN_MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    models_dir = _project_root() / "models"
    candidates.append(models_dir / MODEL_FILENAME)
    candidates.extend(sorted(models_dir.glob("*Qwen3*4B*Q4_K_M.gguf")))

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.suffix.lower() == ".gguf":
            return resolved
    raise QwenAdapterError(
        f"Qwen3 GGUF not found. Set C_ARGUMENTATION_QWEN_MODEL_PATH or place "
        f"{MODEL_FILENAME} under {_project_root() / 'models'}."
    )


def _load_model(model_path: str, n_ctx: int, n_threads: int) -> Any:
    try:
        from llama_cpp import Llama
    except Exception as exc:  # pragma: no cover - environment-specific
        raise QwenAdapterError("llama-cpp-python is not installed or failed to import") from exc

    try:
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=DEFAULT_N_GPU_LAYERS,
            n_threads=n_threads,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - model/runtime-specific
        raise QwenAdapterError(f"Unable to load Qwen3 model at {model_path}: {exc}") from exc


@lru_cache(maxsize=1)
def cached_qwen_model(model_path: str, n_ctx: int = DEFAULT_N_CTX, n_threads: int = DEFAULT_N_THREADS) -> Any:
    """Load one CPU model instance per path/context/thread configuration."""

    return _load_model(model_path, n_ctx, n_threads)


def _strip_thinking(text: str) -> str:
    # Qwen3 may emit a reasoning block even when a concise answer is requested.
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class LocalQwenAdapter:
    """Small, cache-backed local inference facade for C4."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: int = DEFAULT_N_THREADS,
    ) -> None:
        if n_ctx < 1 or n_threads < 1:
            raise ValueError("n_ctx and n_threads must be positive")
        self.model_path = resolve_model_path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads

    @property
    def model(self) -> Any:
        return cached_qwen_model(str(self.model_path), self.n_ctx, self.n_threads)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 128,
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
    ) -> QwenGeneration:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            completion_kwargs: dict[str, Any] = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
            }
            if response_format is not None:
                completion_kwargs["response_format"] = response_format
            response = self.model.create_chat_completion(
                **completion_kwargs,
            )
            choice = response["choices"][0]
            text = _strip_thinking(str(choice["message"]["content"]))
            usage = response.get("usage", {})
            return QwenGeneration(
                text=text,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
        except Exception as exc:  # pragma: no cover - runtime-specific
            raise QwenAdapterError(f"Qwen3 inference failed: {exc}") from exc


def clear_qwen_cache() -> None:
    """Release the cached reference for tests or an explicit lifecycle reset."""

    cached_qwen_model.cache_clear()
