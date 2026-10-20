"""Basic LaBSE + pgvector retrieval for legal chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_TOP_K = 50


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    original_chunk_id: str
    text: str
    metadata: dict[str, Any]
    parent_context: dict[str, Any] | None
    paragraph: str | None
    distance: float
    similarity: float


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be a positive integer")
    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")


def _validate_filter(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string when provided")
    return value.strip()
