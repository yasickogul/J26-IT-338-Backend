"""Page-aware PyMuPDF extraction for C4 judgment PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(slots=True)
class ExtractedJudgment:
    page_texts: list[str]
    full_text: str
    page_count: int
    character_count: int
    extraction_status: str
    extraction_method: str


def extract_document(pdf_path: str | Path, *, low_quality_chars_per_page: int = 40) -> ExtractedJudgment:
    """Extract text page by page and mark likely scanned/low-quality PDFs."""

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    document = pymupdf.open(path)
    try:
        page_texts = [(page.get_text("text") or "").strip() for page in document]
    finally:
        document.close()
    page_count = len(page_texts)
    character_count = sum(len(page) for page in page_texts)
    full_text = "\n\n".join(
        f"[PAGE {index}]\n{text}" for index, text in enumerate(page_texts, start=1) if text
    ).strip()
    minimum = max(100, page_count * low_quality_chars_per_page)
    status = "ocr_required" if page_count == 0 or character_count < minimum else "extracted"
    return ExtractedJudgment(page_texts, full_text, page_count, character_count, status, "pymupdf")


def save_extracted_text(result: ExtractedJudgment, output_path: str | Path) -> Path:
    """Save page-marked text locally for inspection without altering wording."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.full_text, encoding="utf-8")
    return path


def extract_text(pdf_path: str | Path) -> str:
    """Compatibility helper returning the page-marked extracted text."""
    return extract_document(pdf_path).full_text


def extract_directory(pdf_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    source = Path(pdf_dir)
    extracted: dict[str, Path] = {}
    for pdf_path in sorted(source.glob("*.pdf")):
        result = extract_document(pdf_path)
        extracted[pdf_path.name] = save_extracted_text(result, Path(output_dir) / f"{pdf_path.stem}.txt")
    return extracted
