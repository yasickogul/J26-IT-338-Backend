"""Document processing for Stage 2 — PDF/text → cleaned narrative."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from app.components.c1_CaseAnalysis.config import OCR_MIN_CHARS
from app.components.c1_CaseAnalysis.pipeline.preprocessor import (
    clean_text,
    detect_language_hint,
    split_sentences,
)


def extract_pdf_text(path: Path | str) -> str:
    doc = pymupdf.open(path)
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text") or "")
        return "\n".join(parts)
    finally:
        doc.close()


def process_document_text(raw: str) -> dict:
    cleaned = clean_text(raw)
    sentences = split_sentences(cleaned)
    return {
        "text": cleaned,
        "char_count": len(cleaned),
        "sentence_count": len(sentences),
        "sentences": sentences,
        "language_hint": detect_language_hint(cleaned),
        "needs_ocr": len(cleaned) < OCR_MIN_CHARS,
    }


def process_pdf(path: Path | str) -> dict:
    raw = extract_pdf_text(path)
    result = process_document_text(raw)
    result["source"] = "pdf"
    result["path"] = str(path)
    return result
