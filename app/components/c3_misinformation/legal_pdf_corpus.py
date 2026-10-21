"""Prepare and import the reviewed additive C3 legal-PDF batch.

This module keeps the established four-artifact contract.  It uses PyMuPDF
page extraction, the existing C3 Tesseract adapter for image-only pages, the
existing conservative text cleaner, and the configured LaBSE implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from sqlalchemy import text

from app.core.database import DirectSessionLocal, direct_engine

from .corpus_import import (
    IMPORT_AUTHORIZATION,
    import_c3_artifacts,
    validate_c3_artifacts,
)
from .pipeline.labse_query_embedding import (
    MODEL_NAME,
    _get_model,
    embed_texts,
    validate_input_token_limits,
)
from .pipeline.ocr_extractor import (
    ImageProcessingConfig,
    PillowImageProcessor,
    TesseractOCRProvider,
)
from .pipeline.text_preprocessing import clean_text

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "c3_misinformation"
BATCH_NAME = "legal_additions_20261021"
OCR_MIN_CONFIDENCE = 0.45
EMBEDDING_TOKEN_TARGET = 240


@dataclass(frozen=True)
class DocumentSpec:
    source_id: str
    relative_pdf: str
    title: str
    legal_domain: str
    document_type: str
    act_number: str


DOCUMENTS = (
    DocumentSpec("CON-004", "legal_sources/consumer/Source_004/5802.pdf", "Consumer Affairs Authority Act, No. 9 of 2003", "Consumer Rights", "act", "No. 9 of 2003"),
    DocumentSpec("EMP-004", "legal_sources/employment/Source_004/35.pdf", "Industrial Disputes (Amendment) Act, No. 21 of 2008", "Employment", "Amendment Act", "No. 21 of 2008"),
    DocumentSpec("EMP-005", "legal_sources/employment/Source_005/3045.pdf", "Industrial Disputes (Amendment) Act, No. 32 of 1990", "Employment", "Amendment Act", "No. 32 of 1990"),
    DocumentSpec("EMP-006", "legal_sources/employment/Source_006/3080.pdf", "Industrial Disputes (Amendment) Act, No. 25 of 1956", "Employment", "Amendment Act", "No. 25 of 1956"),
    DocumentSpec("EMP-007", "legal_sources/employment/Source_007/3569.pdf", "Industrial Disputes (Amendment) Act, No. 14 of 1957", "Employment", "Amendment Act", "No. 14 of 1957"),
    DocumentSpec("EMP-008", "legal_sources/employment/Source_008/3707.pdf", "Shop and Office Employees (Regulation of Employment and Remuneration) (Amendment) Act, No. 60 of 1957", "Employment", "Amendment Act", "No. 60 of 1957"),
    DocumentSpec("EMP-009", "legal_sources/employment/Source_009/3072.pdf", "Payment of Gratuity (Amendment) Act, No. 41 of 1990", "Employment", "Amendment Act", "No. 41 of 1990"),
    DocumentSpec("EMP-010", "legal_sources/employment/Source_010/3120.pdf", "Payment of Gratuity Act, No. 12 of 1983", "Employment", "act", "No. 12 of 1983"),
    DocumentSpec("PRO-004", "legal_sources/property/Source_004/59.pdf", "Inland Revenue (Amendment) Act, No. 10 of 2007", "Property & Rental", "Amendment Act", "No. 10 of 2007"),
    DocumentSpec("PRO-005", "legal_sources/property/Source_005/2636.pdf", "Inland Revenue (Amendment) Act, No. 19 of 2009", "Property & Rental", "Amendment Act", "No. 19 of 2009"),
    DocumentSpec("PRO-006", "legal_sources/property/Source_006/2723.pdf", "Land Acquisition (Amendment) Act, No. 8 of 1979", "Property & Rental", "Amendment Act", "No. 8 of 1979"),
    DocumentSpec("PRO-007", "legal_sources/property/Source_007/2697.pdf", "National Housing (Amendment) Act, No. 9 of 1978", "Property & Rental", "Amendment Act", "No. 9 of 1978"),
    DocumentSpec("PRO-008", "legal_sources/property/Source_008/2864.pdf", "Code of Intellectual Property (Amendment) Act, No. 30 of 1980", "Property & Rental", "Amendment Act", "No. 30 of 1980"),
    DocumentSpec("PRO-009", "legal_sources/property/Source_009/2873.pdf", "Housing and Town Improvement (Amendment) Act, No. 38 of 1980", "Property & Rental", "Amendment Act", "No. 38 of 1980"),
    DocumentSpec("PRO-010", "legal_sources/property/Source_010/2962.pdf", "National Housing (Amendment) Act, No. 49 of 1981", "Property & Rental", "Amendment Act", "No. 49 of 1981"),
)

DUPLICATE_CANDIDATE = "legal_sources/consumer/Source_005/6225.pdf"
DUPLICATE_ORIGINAL = "legal_sources/consumer/source_001/CON001.pdf"

_PUBLICATION_PREFIXES = (
    "printed at the department of government printing",
    "to be purchased at the government publications bureau",
    "published as a supplement to part ii of the gazette",
    "this act can be downloaded from",
    "printed on the order of government",
    "printed on the orders of government",
    "annual subscription (including bills)",
    "price :",
    "price:",
    "postage :",
    "postage:",
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _configure_tesseract() -> None:
    import pytesseract

    candidate = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if candidate.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(candidate)


def _looks_like_cover(page_number: int, value: str) -> bool:
    if page_number != 1:
        return False
    folded = value.casefold()
    has_body = "be it enacted" in folded or bool(
        re.search(
            r"(?m)^\s*(?:section\s+)?\d+[a-z]?\s*[.\u2014-]",
            value,
            re.IGNORECASE,
        )
    )
    return not has_body


def _clean_page(value: str, *, title: str, ocr: bool) -> str:
    cleaned = clean_text(value, ocr_mode=ocr).replace("\ufffd", " ")
    title_key = " ".join(title.casefold().split())
    raw_lines = [" ".join(line.split()).strip() for line in cleaned.splitlines()]
    # Government Printer headers commonly split the Act title over two or
    # three lines.  Remove only an exact title assembled from the first lines.
    for end in range(1, min(4, len(raw_lines)) + 1):
        leading = " ".join(" ".join(raw_lines[:end]).casefold().split())
        if leading == title_key:
            raw_lines = raw_lines[end:]
            break
    kept: list[str] = []
    for line in raw_lines:
        folded = line.casefold()
        if not line or re.fullmatch(r"[-\u2013\u2014\s]*\d+[-\u2013\u2014\s]*", line):
            continue
        if any(folded.startswith(prefix) for prefix in _PUBLICATION_PREFIXES):
            continue
        if re.fullmatch(
            r"\d+\s*[\u2014-]\s*PL\b.*", line, re.IGNORECASE,
        ) or re.fullmatch(
            r"L[.]D[.]\s*[\u2014-]?.*", line, re.IGNORECASE,
        ):
            continue
        line_key = " ".join(folded.split())
        if line_key == title_key:
            continue
        if sum(character.isalnum() for character in line) < 3:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _token_count(tokenizer: Any, value: str) -> int:
    return len(tokenizer(value, truncation=False, add_special_tokens=True)["input_ids"])


def _split_oversized_unit(
    unit: str, *, prefix: str, tokenizer: Any, target: int,
) -> list[str]:
    words = unit.split()
    parts: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and _token_count(tokenizer, prefix + candidate) > target:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    if any(_token_count(tokenizer, prefix + part) > target for part in parts):
        raise ValueError("A single extracted token exceeds the LaBSE input budget")
    return parts


def _page_chunks(
    value: str, *, title: str, page_number: int, tokenizer: Any,
) -> list[str]:
    prefix = f"{title} | PDF page {page_number} | "
    units: list[str] = []
    for line in value.splitlines():
        if _token_count(tokenizer, prefix + line) <= EMBEDDING_TOKEN_TARGET:
            units.append(line)
        else:
            units.extend(_split_oversized_unit(
                line, prefix=prefix, tokenizer=tokenizer,
                target=EMBEDDING_TOKEN_TARGET,
            ))
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if current and _token_count(tokenizer, prefix + candidate) > EMBEDDING_TOKEN_TARGET:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _section_label(value: str, *, page_number: int, part: int, total: int) -> str:
    match = re.search(r"\bsection\s+(\d+[A-Za-z]?)\b", value, re.IGNORECASE)
    base = f"Section {match.group(1)}" if match else f"PDF page {page_number}"
    page = f"PDF page {page_number}"
    label = base if base == page else f"{base} > {page}"
    return f"{label} > Part {part}" if total > 1 else label


def _extract_document(spec: DocumentSpec, *, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = DATA_ROOT / "legal_documents" / spec.relative_pdf
    if not path.is_file():
        raise FileNotFoundError(path)
    processor = PillowImageProcessor()
    provider = TesseractOCRProvider(psm=6)
    image_config = ImageProcessingConfig(
        max_file_bytes=30_000_000,
        max_decoded_pixels=50_000_000,
        max_processed_pixels=50_000_000,
        improve_contrast=True,
        reduce_noise=False,
        scale_factor=1,
    )
    page_records: list[dict[str, Any]] = []
    blocked_pages: list[int] = []
    chunks: list[dict[str, Any]] = []
    chunk_number = 0
    with pymupdf.open(path) as document:
        for index, page in enumerate(document):
            page_number = index + 1
            extracted = page.get_text("text") or ""
            method = "embedded"
            confidence: float | None = None
            if len(re.sub(r"\s+", "", extracted)) < 80:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                processed = processor.preprocess(
                    pixmap.tobytes("png"), config=image_config,
                )
                ocr_result = provider.extract(processed, languages=("eng",))
                extracted = ocr_result.raw_text
                confidence = ocr_result.confidence
                method = ocr_result.method
            cleaned = _clean_page(extracted, title=spec.title, ocr=method == "tesseract")
            cover = _looks_like_cover(page_number, cleaned)
            alphanumeric = sum(character.isalnum() for character in cleaned)
            excluded_reason = None
            if cover:
                excluded_reason = "cover_or_publication_page"
            elif alphanumeric < 40:
                excluded_reason = "unreadable_or_empty_page"
                blocked_pages.append(page_number)
            elif confidence is not None and confidence < OCR_MIN_CONFIDENCE:
                excluded_reason = "low_ocr_confidence"
                blocked_pages.append(page_number)
            page_records.append({
                "page": page_number,
                "method": method,
                "ocr_confidence": confidence,
                "raw_text": extracted,
                "cleaned_text": cleaned,
                "excluded_reason": excluded_reason,
            })
            if excluded_reason is not None:
                continue
            page_parts = _page_chunks(
                cleaned, title=spec.title, page_number=page_number,
                tokenizer=tokenizer,
            )
            for part, legal_text in enumerate(page_parts, start=1):
                chunk_number += 1
                chunk_id = f"{spec.source_id}-{chunk_number:04d}"
                section = _section_label(
                    legal_text, page_number=page_number, part=part,
                    total=len(page_parts),
                )
                metadata = {
                    "source_id": spec.source_id,
                    "legal_domain": spec.legal_domain,
                    "title": spec.title,
                    "document_type": spec.document_type,
                    "act_number": spec.act_number,
                    "language": "English",
                    "source_file": spec.relative_pdf,
                    "section_label": section,
                    "page_number": page_number,
                    "page_start": page_number,
                    "page_end": page_number,
                    "extraction_method": method,
                }
                embedding_text = f"{spec.title} | {section} | {legal_text}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "original_chunk_id": chunk_id,
                    "text": legal_text,
                    "embedding_text": embedding_text,
                    "metadata": metadata,
                })
    cleaned_root = DATA_ROOT / "cleaned_documents" / Path(spec.relative_pdf).parent
    cleaned_root.mkdir(parents=True, exist_ok=True)
    (cleaned_root / "pages.json").write_text(
        json.dumps(page_records, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (cleaned_root / "extracted.txt").write_text(
        "\n\n".join(
            f"[PDF page {page['page']}]\n{page['cleaned_text']}"
            for page in page_records if page["excluded_reason"] is None
        ),
        encoding="utf-8",
    )
    if blocked_pages:
        chunks = []
    return chunks, {
        **asdict(spec),
        "pdf_sha256": _hash(path),
        "pages": len(page_records),
        "embedded_pages": sum(page["method"] == "embedded" for page in page_records),
        "ocr_pages": sum(page["method"] == "tesseract" for page in page_records),
        "excluded_cover_pages": sum(
            page["excluded_reason"] == "cover_or_publication_page" for page in page_records
        ),
        "blocked_pages": blocked_pages,
        "chunks": len(chunks),
        "status": "blocked" if blocked_pages or not chunks else "prepared",
    }


def prepare_artifacts() -> tuple[dict[str, Path], dict[str, Any]]:
    _configure_tesseract()
    duplicate = DATA_ROOT / "legal_documents" / DUPLICATE_CANDIDATE
    original = DATA_ROOT / "legal_documents" / DUPLICATE_ORIGINAL
    if _hash(duplicate) != _hash(original):
        raise ValueError("Expected duplicate candidate no longer matches the imported source PDF")

    encoder = _get_model()
    tokenizer = encoder.tokenizer
    all_chunks: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for position, spec in enumerate(DOCUMENTS, start=1):
        chunks, report = _extract_document(spec, tokenizer=tokenizer)
        documents.append(report)
        all_chunks.extend(chunks)
        print(
            f"prepared {position}/{len(DOCUMENTS)} {spec.source_id}: "
            f"pages={report['pages']} chunks={report['chunks']} status={report['status']}",
            flush=True,
        )

    blocked = [item for item in documents if item["status"] != "prepared"]
    if blocked:
        names = [item["source_id"] for item in blocked]
        raise ValueError(f"Unreadable content blocked documents: {names}")
    ids = [item["chunk_id"] for item in all_chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated duplicate chunk identifiers")
    embedding_texts = [item["embedding_text"] for item in all_chunks]
    token_counts = validate_input_token_limits(embedding_texts, encoder=encoder)
    vectors = embed_texts(embedding_texts, encoder=encoder).astype(np.float32, copy=False)

    chunks_path = DATA_ROOT / "chunks" / f"{BATCH_NAME}.json"
    inputs_path = DATA_ROOT / "embeddings" / f"{BATCH_NAME}_inputs.json"
    vectors_path = DATA_ROOT / "embeddings" / f"{BATCH_NAME}_labse.npz"
    manifest_path = DATA_ROOT / "manifests" / f"{BATCH_NAME}_manifest.json"
    report_path = DATA_ROOT / "manifests" / f"{BATCH_NAME}_report.json"
    for path in (chunks_path, inputs_path, vectors_path, manifest_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps({"chunks": all_chunks}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    prepared = {
        "model_name": MODEL_NAME,
        "dimension": 768,
        "tokenizer_settings": {
            "add_special_tokens": True,
            "truncation": False,
            "max_seq_length": encoder.max_seq_length,
        },
        "derived_input_count": len(all_chunks),
        "inputs": all_chunks,
    }
    inputs_path.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    np.savez(
        vectors_path,
        embeddings=vectors,
        chunk_ids=np.asarray(ids, dtype=f"<U{max(map(len, ids))}"),
    )
    manifest = {
        "model_name": MODEL_NAME,
        "dimension": 768,
        "normalization": "unit-length (normalize_embeddings=True)",
        "input_file_sha256": _hash(inputs_path),
        "original_corpus_sha256": _hash(chunks_path),
        "count": len(all_chunks),
        "batch_name": BATCH_NAME,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    report = {
        "batch_name": BATCH_NAME,
        "documents": documents,
        "skipped_duplicates": [{
            "file": DUPLICATE_CANDIDATE,
            "duplicate_of": DUPLICATE_ORIGINAL,
            "sha256": _hash(duplicate),
        }],
        "chunks": len(all_chunks),
        "embeddings": int(vectors.shape[0]),
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "token_limit": encoder.max_seq_length,
        "ocr_pages": sum(item["ocr_pages"] for item in documents),
        "embedded_pages": sum(item["embedded_pages"] for item in documents),
        "vector_sha256": _hash(vectors_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    paths = {
        "data_root": DATA_ROOT,
        "manifest_path": manifest_path,
        "input_path": inputs_path,
        "corpus_path": chunks_path,
        "vector_path": vectors_path,
    }
    _, _, validation = validate_c3_artifacts(**paths)
    print(json.dumps({"prepared": report, "validation": validation}, indent=2), flush=True)
    return paths, report


async def import_and_verify(paths: dict[str, Path]) -> dict[str, Any]:
    if DirectSessionLocal is None or direct_engine is None:
        raise RuntimeError("DATABASE_URL / DATABASE_URL_DIRECT is not configured")
    async with DirectSessionLocal() as session:
        receipt = await import_c3_artifacts(
            session=session,
            authorization=IMPORT_AUTHORIZATION,
            incremental=True,
            **paths,
        )
    async with direct_engine.connect() as connection:
        counts = (await connection.execute(text(
            "SELECT count(*) AS sources, "
            "(SELECT count(*) FROM c3.legal_chunks) AS chunks, "
            "(SELECT count(*) FROM c3.legal_chunks WHERE embedding IS NOT NULL) AS embeddings "
            "FROM c3.legal_sources"
        ))).mappings().one()
        imported = (await connection.execute(text(
            "SELECT source_id, count(*) AS chunks, "
            "count(embedding) AS embeddings "
            "FROM c3.legal_chunks WHERE source_id = ANY(:source_ids) "
            "GROUP BY source_id ORDER BY source_id"
        ), {"source_ids": [spec.source_id for spec in DOCUMENTS]})).mappings().all()
    return {
        "receipt": receipt,
        "database_counts": dict(counts),
        "imported_sources": [dict(row) for row in imported],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--import-db", action="store_true",
        help="After preparation and validation, add the batch to the configured C3 corpus",
    )
    args = parser.parse_args()
    paths, _ = prepare_artifacts()
    if args.import_db:
        result = asyncio.run(import_and_verify(paths))
        print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
