"""Conservative, deterministic text normalization for legal claim input."""

from __future__ import annotations

import unicodedata


def _normalize_page(page: str) -> list[str]:
    lines: list[str] = []
    for raw_line in page.split("\n"):
        line = " ".join(raw_line.split())
        if line or not lines or lines[-1]:
            lines.append(line)

    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def clean_text(
    text: str,
    *,
    remove_page_boundary_duplicates: bool = False,
    ocr_mode: bool = False,
) -> str:
    """Normalize spacing and NFC without guessing at uncertain OCR errors.

    OCR mode enables only exact page-boundary duplicate removal. Typed legal
    claims preserve repeated lines because intent cannot be inferred safely.
    No uncertain spelling or punctuation correction is attempted.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(remove_page_boundary_duplicates, bool):
        raise TypeError("remove_page_boundary_duplicates must be a boolean")
    if not isinstance(ocr_mode, bool):
        raise TypeError("ocr_mode must be a boolean")

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.removeprefix("\ufeff")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\t", "\n", "\f", "\r"}
        or unicodedata.category(character) not in {"Cc", "Cs"}
    )
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    pages = [_normalize_page(page) for page in normalized.split("\f")]
    if remove_page_boundary_duplicates or ocr_mode:
        for index in range(1, len(pages)):
            previous = pages[index - 1]
            current = pages[index]
            if previous and current and previous[-1] == current[0]:
                current.pop(0)

    lines = [line for page in pages for line in page]
    return "\n".join(lines).strip()
