"""Small, side-effect-free readers for C3 runtime feature flags."""

from __future__ import annotations

import os

from .agents.gemini_adapter import DEFAULT_MODEL, GeminiAdapter
from .agents.qwen_adapter import (
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    LocalQwenAdapter,
)


def boolean_env(name: str, *, default: bool = False) -> bool:
    """Read a strict boolean environment value without changing its default."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("environment variable name must be nonblank")
    if not isinstance(default, bool):
        raise TypeError("default must be a boolean")

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = raw_value.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise RuntimeError(f"{name} must be a boolean value.")


def language_detection_enabled() -> bool:
    """Return the established opt-in statistical language-detection flag."""
    return boolean_env("COMPNET_LANGUAGE_DETECTION_ENABLED")


def claim_extraction_enabled() -> bool:
    """Return the established opt-in worker claim-extraction flag."""
    return boolean_env("COMPNET_CLAIM_EXTRACTION_ENABLED")


def llm_provider() -> str:
    """Return the explicitly selected C3 structured-output provider."""
    value = os.getenv("C3_LLM_PROVIDER", "gemini").strip().casefold()
    if value not in {"gemini", "qwen"}:
        raise RuntimeError("C3_LLM_PROVIDER must be gemini or qwen")
    return value


def qwen_base_url() -> str:
    value = os.getenv("C3_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).strip()
    if not value:
        raise RuntimeError("C3_QWEN_BASE_URL must be nonblank")
    return value


def qwen_model() -> str:
    value = os.getenv("C3_QWEN_MODEL", DEFAULT_QWEN_MODEL).strip()
    if not value:
        raise RuntimeError("C3_QWEN_MODEL must be nonblank")
    return value


def create_structured_adapter():
    """Build the selected C3 adapter without making a network request."""
    if llm_provider() == "qwen":
        return LocalQwenAdapter(base_url=qwen_base_url(), model=qwen_model())
    return GeminiAdapter(model=DEFAULT_MODEL)


__all__ = [
    "boolean_env",
    "claim_extraction_enabled",
    "create_structured_adapter",
    "language_detection_enabled",
    "llm_provider",
    "qwen_base_url",
    "qwen_model",
]
