"""Text cleaning / normalization for Stage 2."""

from __future__ import annotations

import re

_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


def clean_text(raw: str | None) -> str:
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse horizontal whitespace per line, then trim blank runs.
    text = "\n".join(_WS_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def detect_language_hint(text: str) -> str:
    """Lightweight hint only — not a full language ID model."""
    if re.search(r"[\u0D80-\u0DFF]", text):
        return "si"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    return "en"
