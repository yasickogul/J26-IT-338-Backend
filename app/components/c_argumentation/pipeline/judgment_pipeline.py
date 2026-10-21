"""C4 judgment download, extraction, persistence, and chunking orchestration."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, or_, select

from app.core.database import DirectSessionLocal, direct_engine

from ..models import CaseJudgment, CourtCase, JudgmentChunk
from .pdf_downloader import download_pdf
from .text_chunker import chunk_pages
from .text_extractor import extract_document, save_extracted_text


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = COMPONENT_ROOT / "data" / "pdfs"
EXTRACTED_DIR = COMPONENT_ROOT / "data" / "extracted"
PROGRESS_DIR = COMPONENT_ROOT / "data" / "progress"


async def _find_case(case_id: UUID | None = None) -> CourtCase:
    if DirectSessionLocal is None:
        raise RuntimeError("DATABASE_URL/DATABASE_URL_DIRECT is not configured")
    async with DirectSessionLocal() as session:
        query = select(CourtCase).where(CourtCase.pdf_url.is_not(None)).order_by(CourtCase.created_at, CourtCase.id)
        if case_id is not None:
            query = select(CourtCase).where(CourtCase.id == case_id)
        case = (await session.execute(query.limit(1))).scalars().first()
        if case is None:
            raise LookupError("No C4 court case with a PDF URL was found")
        return case


async def process_one_judgment(
    case_id: UUID | None = None,
    *,
    pdf_dir: str | Path = PDF_DIR,
    extracted_dir: str | Path = EXTRACTED_DIR,
) -> dict[str, Any]:
    """Process exactly one case and atomically replace its chunk set."""

    if direct_engine is None or DirectSessionLocal is None:
        raise RuntimeError("DATABASE_URL/DATABASE_URL_DIRECT is not configured")
    case = await _find_case(case_id)
    pdf_path = Path(pdf_dir) / f"{case.source_document_key}.pdf"
    text_path = Path(extracted_dir) / f"{case.source_document_key}.txt"

    async with DirectSessionLocal() as session:
        existing = (await session.execute(select(CaseJudgment).where(CaseJudgment.case_id == case.id))).scalars().first()
        if existing and existing.extraction_status == "extracted" and pdf_path.exists() and text_path.exists():
            return {"case_id": str(case.id), "case_number": case.case_number, "status": "skipped", "reason": "already_extracted"}

    download = await asyncio.to_thread(
        download_pdf,
        source_document_key=case.source_document_key,
        pdf_url=case.pdf_url or "",
        output_dir=pdf_dir,
    )
    if download.path is None:
        async with DirectSessionLocal() as session:
            async with session.begin():
                judgment = (await session.execute(select(CaseJudgment).where(CaseJudgment.case_id == case.id))).scalars().first()
                if judgment is None:
                    judgment = CaseJudgment(case_id=case.id)
                    session.add(judgment)
                judgment.extraction_status = download.status
                judgment.extraction_method = "download"
        return {"case_id": str(case.id), "case_number": case.case_number, "status": download.status, "error": download.error}

    extracted = await asyncio.to_thread(extract_document, download.path)
    await asyncio.to_thread(save_extracted_text, extracted, text_path)
    chunks = chunk_pages(extracted.page_texts) if extracted.extraction_status == "extracted" else []

    async with DirectSessionLocal() as session:
        async with session.begin():
            judgment = (await session.execute(select(CaseJudgment).where(CaseJudgment.case_id == case.id).with_for_update())).scalars().first()
            if judgment is None:
                judgment = CaseJudgment(case_id=case.id)
                session.add(judgment)
                await session.flush()
            judgment.full_text = extracted.full_text
            judgment.page_count = extracted.page_count
            judgment.extraction_status = extracted.extraction_status
            judgment.extraction_method = extracted.extraction_method
            judgment.extracted_at = datetime.now(timezone.utc)
            await session.execute(delete(JudgmentChunk).where(JudgmentChunk.judgment_id == judgment.id))
            if extracted.extraction_status == "extracted":
                session.add_all([
                    JudgmentChunk(
                        judgment_id=judgment.id,
                        chunk_index=chunk.chunk_index,
                        start_page=chunk.start_page,
                        end_page=chunk.end_page,
                        chunk_text=chunk.chunk_text,
                    )
                    for chunk in chunks
                ])

    return {
        "case_id": str(case.id),
        "case_number": case.case_number,
        "source_document_key": case.source_document_key,
        "download_status": download.status,
        "status": extracted.extraction_status,
        "pdf_path": str(download.path),
        "text_path": str(text_path),
        "page_count": extracted.page_count,
        "character_count": extracted.character_count,
        "chunk_count": len(chunks),
        "sample": extracted.full_text[:500],
    }


async def process_batch(*, limit: int = 25, concurrency: int = 2) -> dict[str, Any]:
    """Process one bounded, resumable batch and save its progress report."""

    if limit < 1 or concurrency < 1:
        raise ValueError("limit and concurrency must be at least 1")
    async with DirectSessionLocal() as session:
        cases = (await session.execute(
            select(CourtCase.id)
            .outerjoin(CaseJudgment, CaseJudgment.case_id == CourtCase.id)
            .where(
                CourtCase.pdf_url.is_not(None),
                or_(CaseJudgment.id.is_(None), CaseJudgment.extraction_status != "extracted"),
            )
            .distinct()
            .order_by(CourtCase.id)
            .limit(limit)
        )).scalars().all()
    semaphore = asyncio.Semaphore(concurrency)

    async def run(case_id: UUID) -> dict[str, Any]:
        async with semaphore:
            try:
                return await process_one_judgment(case_id)
            except Exception as exc:  # continue the batch without exposing credentials
                return {"case_id": str(case_id), "status": "failed", "error": type(exc).__name__}

    results = await asyncio.gather(*(run(case_id) for case_id in cases))
    counts = Counter(str(result.get("status", "failed")) for result in results)
    async with DirectSessionLocal() as session:
        cumulative_judgments = await session.scalar(select(func.count()).select_from(CaseJudgment))
        cumulative_chunks = await session.scalar(select(func.count()).select_from(JudgmentChunk))
    report: dict[str, Any] = {
        "batch_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempted": len(cases),
        "completed": counts.get("extracted", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "ocr_required": counts.get("ocr_required", 0),
        "access_restricted": counts.get("access_restricted", 0),
        "downloaded": sum(result.get("download_status") == "downloaded" for result in results),
        "total_pages": sum(int(result.get("page_count") or 0) for result in results),
        "chunks_generated": sum(int(result.get("chunk_count") or 0) for result in results),
        "cumulative_case_judgments": int(cumulative_judgments or 0),
        "cumulative_judgment_chunks": int(cumulative_chunks or 0),
    }
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    progress_path = PROGRESS_DIR / f"batch_{timestamp}.json"
    progress_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["progress_report"] = str(progress_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="C4 judgment pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single-test", action="store_true", help="process one database-selected judgment")
    mode.add_argument("--batch", action="store_true", help="process a bounded resumable batch")
    parser.add_argument("--case-id", type=UUID)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.single_test:
        print(json.dumps(asyncio.run(process_one_judgment(args.case_id)), default=str, indent=2))
    else:
        print(json.dumps(asyncio.run(process_batch(limit=args.limit, concurrency=args.concurrency)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
