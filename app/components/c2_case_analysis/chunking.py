"""Chunking (notebook cell 6 `chunk_document` and cell 14 `chunk_for_retrieval`).

NOTE - one deliberate fix versus the notebook:
In the notebook, once the window reached the end of the text the loop kept advancing
`start` by a single character and re-emitted almost identical chunks (a 30k-character
document produced ~580 chunks => ~580 LLM calls). Here, the final window is emitted
once and the loop stops. Every earlier chunk boundary and overlap is identical to the
notebook's.
"""
import re


def _split(text, max_chars, overlap_chars):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))

        if end >= len(text):  # final window: take everything that is left, then stop
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # try to break on a sentence boundary near the end
        boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end
        chunk = text[start:boundary + 1].strip()
        if chunk:
            chunks.append(chunk)
        start = max(boundary + 1 - overlap_chars, start + 1)
    return chunks


def chunk_document(text, max_chars=9000, overlap_chars=500):
    """Large chunks used for LLM category extraction."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]
    return _split(text, max_chars, overlap_chars)


def chunk_for_retrieval(text, max_chars=1000, overlap_chars=150):
    """Small chunks used when building the knowledge base."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    return _split(text, max_chars, overlap_chars)
