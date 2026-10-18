#!/usr/bin/env python3
"""Ingest downloaded CoA judgment PDFs into shared.legal_documents / legal_chunks.

Usage (from J26-IT-338-Backend/):
  python -m scripts.ingest_judgments
  python -m scripts.ingest_judgments --limit 50
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pymupdf
from sqlalchemy import select

from app.core.database import DirectSessionLocal, direct_engine
from app.shared.corpus.ingest import citation_exists, insert_judgment, parse_decision_date
from app.shared.corpus.models import LegalDocument
from app.shared.corpus.schema import ensure_corpus_schema
from app.shared.utils.text import (
    clean_parties,
    court_from_case_number,
    sanitize_case_number,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = BACKEND_ROOT.parent
DEFAULT_CIVIL_CSV = APP_ROOT / "Dataset" / "court_of_appeal_civil_cases.csv"
DEFAULT_RAW_DIR = BACKEND_ROOT / "data" / "judgments" / "raw"
OCR_MIN_CHARS = 200


def load_civil_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("case_type") or "").strip().lower() != "civil":
                continue
            case_number = (row.get("case_number") or "").strip()
            if not case_number or case_number in seen:
                continue
            seen.add(case_number)
            rows.append(row)
    return rows


def extract_pdf_text(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text") or "")
        return "\n".join(parts).strip()
    finally:
        doc.close()


def title_from_parties(parties: str, case_number: str) -> str:
    if parties:
        # Keep titles concise for display / indexing.
        return parties if len(parties) <= 240 else parties[:237].rstrip() + "..."
    return case_number


async def run(args: argparse.Namespace) -> int:
    if direct_engine is None or DirectSessionLocal is None:
        print("ERROR: DATABASE_URL / DATABASE_URL_DIRECT is not set", flush=True)
        return 1

    csv_path: Path = args.csv or DEFAULT_CIVIL_CSV
    raw_dir: Path = args.raw_dir or DEFAULT_RAW_DIR

    print(f"CSV     : {csv_path} ({'exists' if csv_path.exists() else 'MISSING'})", flush=True)
    print(f"PDF dir : {raw_dir} ({'exists' if raw_dir.exists() else 'MISSING'})", flush=True)

    rows = load_civil_rows(csv_path)
    rows_read = len(rows)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    print("Ensuring shared corpus schema (direct Neon connection)…", flush=True)
    await ensure_corpus_schema(direct_engine)
    print("Schema ready.", flush=True)

    stats = {
        "rows_read": rows_read,
        "rows_considered": len(rows),
        "pdfs_on_disk": 0,
        "ingested": 0,
        "chunks": 0,
        "needs_ocr": 0,
        "duplicates_csv": 0,  # already handled in load
        "duplicates_db": 0,
        "missing_pdf": 0,
        "errors": 0,
    }
    ocr_cases: list[str] = []

    async with DirectSessionLocal() as session:
        # Fast path: existing citations already in DB
        existing = set(
            (
                await session.execute(
                    select(LegalDocument.citation).where(LegalDocument.source_type == "judgment")
                )
            )
            .scalars()
            .all()
        )
        existing = {c for c in existing if c}

        for row in rows:
            case_number = row["case_number"].strip()
            pdf_path = raw_dir / f"{sanitize_case_number(case_number)}.pdf"
            if not pdf_path.exists():
                stats["missing_pdf"] += 1
                continue
            stats["pdfs_on_disk"] += 1

            if case_number in existing:
                stats["duplicates_db"] += 1
                continue

            try:
                full_text = extract_pdf_text(pdf_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR extract {case_number}: {exc}")
                stats["errors"] += 1
                continue

            if len(full_text) < OCR_MIN_CHARS:
                stats["needs_ocr"] += 1
                ocr_cases.append(case_number)
                print(f"  OCR  {case_number} ({len(full_text)} chars) — skipped")
                continue

            parties = clean_parties(row.get("parties"))
            court, prefix = court_from_case_number(case_number)
            decision_date = parse_decision_date(row.get("date_decided"))
            metadata = {
                "judge": (row.get("judge") or "").strip(),
                "keywords": (row.get("keywords") or "").strip(),
                "legislation": (row.get("legislation") or "").strip(),
                "pdf_url": (row.get("pdf_url") or "").strip(),
                "case_number_prefix": prefix,
            }

            # Double-check race / prior insert in this session
            if await citation_exists(session, case_number):
                stats["duplicates_db"] += 1
                existing.add(case_number)
                continue

            try:
                _doc, n_chunks = await insert_judgment(
                    session,
                    title=title_from_parties(parties, case_number),
                    citation=case_number,
                    court=court,
                    case_type="civil",
                    decision_date=decision_date,
                    source_url=(row.get("pdf_url") or "").strip() or None,
                    full_text=full_text,
                    metadata=metadata,
                )
                await session.commit()
                existing.add(case_number)
                stats["ingested"] += 1
                stats["chunks"] += n_chunks
                print(f"  OK   {case_number}  chunks={n_chunks}  chars={len(full_text)}")
            except Exception as exc:  # noqa: BLE001
                await session.rollback()
                print(f"  ERROR insert {case_number}: {exc}")
                stats["errors"] += 1

    print("\n=== Ingest summary ===")
    print(f"{'metric':<22} {'count':>8}")
    print("-" * 32)
    print(f"{'rows_read':<22} {stats['rows_read']:>8}")
    print(f"{'rows_considered':<22} {stats['rows_considered']:>8}")
    print(f"{'pdfs_on_disk':<22} {stats['pdfs_on_disk']:>8}")
    print(f"{'ingested':<22} {stats['ingested']:>8}")
    print(f"{'chunks_created':<22} {stats['chunks']:>8}")
    print(f"{'needs_ocr':<22} {stats['needs_ocr']:>8}")
    print(f"{'duplicates_db':<22} {stats['duplicates_db']:>8}")
    print(f"{'missing_pdf':<22} {stats['missing_pdf']:>8}")
    print(f"{'errors':<22} {stats['errors']:>8}")
    if ocr_cases:
        print("\nFlagged for OCR:")
        for cn in ocr_cases:
            print(f"  - {cn}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest CoA judgment PDFs into Neon")
    parser.add_argument("--limit", type=int, default=0, help="Only consider first N CSV rows (0=all)")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args()

    import asyncio

    return asyncio.run(run(args))


if __name__ == "__main__":
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    raise SystemExit(main())
