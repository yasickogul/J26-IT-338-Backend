"""Persistence for c1 case intakes / extractions."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import selectinload

from app.components.c1_CaseAnalysis.models import CaseIntake, Extraction

C1_DDL = """
CREATE SCHEMA IF NOT EXISTS c1;

CREATE TABLE IF NOT EXISTS c1.case_intakes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  status TEXT NOT NULL DEFAULT 'queued',
  dispute_category TEXT,
  filename TEXT,
  mime TEXT,
  storage_path TEXT,
  intake_json JSONB DEFAULT '{}',
  raw_text TEXT,
  error TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS c1.extractions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  intake_id UUID NOT NULL UNIQUE REFERENCES c1.case_intakes(id) ON DELETE CASCADE,
  structured_json JSONB DEFAULT '{}',
  confidence DOUBLE PRECISION,
  model_version TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
"""


async def ensure_c1_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        for stmt in C1_DDL.strip().split(";"):
            sql = stmt.strip()
            if sql:
                await conn.execute(text(sql))


async def create_intake(
    session: AsyncSession,
    *,
    dispute_category: str,
    intake_json: dict[str, Any],
    status: str = "queued",
    filename: str | None = None,
    mime: str | None = None,
    storage_path: str | None = None,
    raw_text: str | None = None,
) -> CaseIntake:
    row = CaseIntake(
        dispute_category=dispute_category,
        intake_json=intake_json,
        status=status,
        filename=filename,
        mime=mime,
        storage_path=storage_path,
        raw_text=raw_text,
    )
    session.add(row)
    await session.flush()
    return row


async def get_intake(session: AsyncSession, intake_id: uuid.UUID) -> CaseIntake | None:
    result = await session.execute(
        select(CaseIntake)
        .options(selectinload(CaseIntake.extraction))
        .where(CaseIntake.id == intake_id)
    )
    return result.scalar_one_or_none()


async def save_extraction(
    session: AsyncSession,
    *,
    intake: CaseIntake,
    structured_json: dict[str, Any],
    model_version: str,
    confidence: float | None,
    raw_text: str,
) -> Extraction:
    intake.status = "done"
    intake.raw_text = raw_text
    intake.error = None
    extraction = Extraction(
        intake_id=intake.id,
        structured_json=structured_json,
        model_version=model_version,
        confidence=confidence,
    )
    session.add(extraction)
    await session.flush()
    return extraction


async def mark_failed(session: AsyncSession, intake: CaseIntake, error: str) -> None:
    intake.status = "failed"
    intake.error = error
