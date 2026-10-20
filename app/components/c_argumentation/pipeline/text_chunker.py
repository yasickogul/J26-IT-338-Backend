"""Page-aware legal passage chunking without embeddings."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class JudgmentChunk:
    chunk_index: int
    chunk_text: str
    start_page: int
    end_page: int


def chunk_pages(
    page_texts: Sequence[str],
    *,
    target_tokens: int = 650,
    overlap_tokens: int = 80,
) -> list[JudgmentChunk]:
    """Build approximately 500–800-token chunks while retaining page bounds."""

    if target_tokens < 100 or overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("target_tokens must be >= 100 and overlap_tokens must be smaller")
    units: list[tuple[str, int]] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page_text) if part.strip()]
        for paragraph in paragraphs:
            words = paragraph.split()
            for start in range(0, len(words), target_tokens):
                units.append((" ".join(words[start : start + target_tokens]), page_number))

    chunks: list[JudgmentChunk] = []
    current: list[tuple[str, int]] = []
    current_words = 0
    for unit_text, page in units:
        words = unit_text.split()
        if current and current_words + len(words) > target_tokens:
            chunks.append(_make_chunk(len(chunks), current))
            overlap = " ".join(current[-1][0].split()[-overlap_tokens:]) if overlap_tokens else ""
            current = [(overlap, current[-1][1])] if overlap else []
            current_words = len(overlap.split())
        current.append((unit_text, page))
        current_words += len(words)
    if current:
        chunks.append(_make_chunk(len(chunks), current))
    return chunks


def _make_chunk(index: int, units: list[tuple[str, int]]) -> JudgmentChunk:
    return JudgmentChunk(
        chunk_index=index,
        chunk_text="\n\n".join(text for text, _page in units if text),
        start_page=units[0][1],
        end_page=units[-1][1],
    )


def chunk_text(text: str, *, target_tokens: int = 650, overlap_tokens: int = 80) -> list[str]:
    """Compatibility helper for callers that do not have page metadata."""
    return [chunk.chunk_text for chunk in chunk_pages([text], target_tokens=target_tokens, overlap_tokens=overlap_tokens)]
