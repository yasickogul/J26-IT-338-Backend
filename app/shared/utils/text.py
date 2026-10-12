"""Shared text helpers for corpus ingestion."""

from __future__ import annotations

import re


_VIEW_MORE_RE = re.compile(r"(?i)\bview\s*more\b")
_WS_RE = re.compile(r"\s+")


def clean_parties(raw: str | None) -> str:
    """Strip duplicated party lines and trailing 'View More'; collapse whitespace."""
    if not raw:
        return ""
    text = _VIEW_MORE_RE.sub("", raw)
    text = _WS_RE.sub(" ", text).strip(" ,;\n\t")

    # Exact / near-exact half duplication common in the CoA CSV export.
    n = len(text)
    if n >= 20:
        for split_at in range(n // 2, max(n // 3, 10), -1):
            left = text[:split_at].strip()
            right = text[split_at:].strip()
            if left and left == right:
                return left

    # Fallback: if the string contains itself twice back-to-back after normalize.
    half = n // 2
    left, right = text[:half].strip(), text[half:].strip()
    if left and left == right:
        return left
    return text


def chunk_text(
    text: str,
    *,
    target_tokens: int = 400,
    overlap_ratio: float = 0.15,
) -> list[str]:
    """Split text into ~300–500 token chunks with overlap (word ≈ token proxy)."""
    words = text.split()
    if not words:
        return []

    chunk_size = max(50, target_tokens)
    overlap = max(1, int(chunk_size * overlap_ratio))
    step = max(1, chunk_size - overlap)

    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def sanitize_case_number(case_number: str) -> str:
    """Turn CA/RII/0021/2024 into a filesystem-safe stem."""
    cleaned = re.sub(r"[^\w.\-]+", "_", (case_number or "").strip())
    return cleaned.strip("_") or "unknown"


def court_from_case_number(case_number: str) -> tuple[str, str]:
    """Return (court_name, case_number_prefix) from e.g. CA/RII/0021/2024."""
    parts = [p for p in (case_number or "").split("/") if p]
    prefix = "/".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")
    if prefix.upper().startswith("CA") or (parts and parts[0].upper() == "CA"):
        return "Court of Appeal", prefix
    return "Unknown", prefix
