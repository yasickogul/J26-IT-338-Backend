"""Ingest helpers for shared.legal_documents / shared.legal_chunks."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.corpus.models import LegalChunk, LegalDocument
from app.shared.utils.text import chunk_text


async def citation_exists(session: AsyncSession, citation: str) -> bool:
    result = await session.execute(
        select(LegalDocument.id).where(LegalDocument.citation == citation).limit(1)
    )
    return result.scalar_one_or_none() is not None


async def insert_judgment(
    session: AsyncSession,
    *,
    title: str,
    citation: str,
    court: str,
    case_type: str,
    decision_date: Optional[date],
    source_url: Optional[str],
    full_text: str,
    metadata: dict[str, Any],
    language: str = "en",
) -> tuple[LegalDocument, int]:
    """Insert one judgment + chunks. embedding left NULL. Returns (doc, chunk_count)."""
    doc = LegalDocument(
        source_type="judgment",
        title=title,
        citation=citation,
        court=court,
        case_type=case_type,
        decision_date=decision_date,
        language=language,
        source_url=source_url,
        full_text=full_text,
        metadata_=metadata,
    )
    session.add(doc)
    await session.flush()  # assign doc.id

    pieces = chunk_text(full_text, target_tokens=400, overlap_ratio=0.15)
    for idx, piece in enumerate(pieces):
        session.add(
            LegalChunk(
                document_id=doc.id,
                chunk_index=idx,
                section=None,
                text=piece,
                embedding=None,
                metadata_={},
            )
        )
    return doc, len(pieces)


def parse_decision_date(raw: str | None) -> Optional[date]:
    if not raw:
        return None
    value = raw.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None
