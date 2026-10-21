"""Incremental persistence for C4 judgment chunk embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select, update

from app.core.database import DirectSessionLocal

from ..models import CaseJudgment, JudgmentChunk
from .c4_embedding_service import MODEL_ID, embedding_source_hash, embed_texts


@dataclass(frozen=True, slots=True)
class EmbeddingCandidate:
    chunk_id: UUID
    chunk_text: str
    source_hash: str


@dataclass(frozen=True, slots=True)
class EmbeddingRunReport:
    attempted: int
    embedded: int
    skipped: int
    failed: int


async def _pending_candidates(limit: int | None) -> tuple[list[EmbeddingCandidate], int]:
    if DirectSessionLocal is None:
        raise RuntimeError("DATABASE_URL/DATABASE_URL_DIRECT is not configured")

    async with DirectSessionLocal() as session:
        rows = (await session.execute(
            select(
                JudgmentChunk.id,
                JudgmentChunk.chunk_text,
                JudgmentChunk.embedding_model,
                JudgmentChunk.embedding_status,
                JudgmentChunk.embedding_source_hash,
            )
            .join(CaseJudgment, CaseJudgment.id == JudgmentChunk.judgment_id)
            .where(CaseJudgment.extraction_status == "extracted")
            .order_by(JudgmentChunk.id)
        )).all()

    candidates: list[EmbeddingCandidate] = []
    skipped = 0
    for chunk_id, chunk_text, model_id, status, source_hash in rows:
        expected_hash = embedding_source_hash(chunk_text)
        if model_id == MODEL_ID and status == "ready" and source_hash == expected_hash:
            skipped += 1
            continue
        candidates.append(EmbeddingCandidate(chunk_id, chunk_text, expected_hash))
        if limit is not None and len(candidates) >= limit:
            break
    return candidates, skipped


async def _set_processing(candidates: list[EmbeddingCandidate]) -> None:
    if not candidates:
        return
    async with DirectSessionLocal() as session:
        async with session.begin():
            for candidate in candidates:
                await session.execute(
                    update(JudgmentChunk)
                    .where(JudgmentChunk.id == candidate.chunk_id)
                    .values(
                        embedding_model=MODEL_ID,
                        embedding_status="processing",
                        embedding_source_hash=candidate.source_hash,
                    )
                )


async def _save_results(
    candidates: list[EmbeddingCandidate],
    vectors: dict[UUID, list[float]],
) -> None:
    async with DirectSessionLocal() as session:
        async with session.begin():
            generated_at = datetime.now(timezone.utc)
            for candidate in candidates:
                vector = vectors.get(candidate.chunk_id)
                values: dict[str, Any] = {
                    "embedding_model": MODEL_ID,
                    "embedding_source_hash": candidate.source_hash,
                    "embedding_generated_at": generated_at,
                    "embedding_status": "ready" if vector is not None else "failed",
                }
                if vector is not None:
                    values["embedding"] = vector
                await session.execute(
                    update(JudgmentChunk)
                    .where(JudgmentChunk.id == candidate.chunk_id)
                    .values(**values)
                )


async def embed_pending_chunks(
    *,
    limit: int | None = None,
    batch_size: int = 16,
    model: Any | None = None,
) -> EmbeddingRunReport:
    """Embed a bounded, resumable set of extracted chunks."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided")
    candidates, skipped = await _pending_candidates(limit)
    if not candidates:
        return EmbeddingRunReport(0, 0, skipped, 0)

    await _set_processing(candidates)
    vectors: dict[UUID, list[float]] = {}
    failed = 0
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        try:
            encoded = embed_texts([candidate.chunk_text for candidate in batch], model=model, batch_size=batch_size)
            vectors.update({candidate.chunk_id: vector for candidate, vector in zip(batch, encoded, strict=True)})
        except Exception:
            for candidate in batch:
                try:
                    vector = embed_texts([candidate.chunk_text], model=model, batch_size=1)[0]
                    vectors[candidate.chunk_id] = vector
                except Exception:
                    failed += 1
        await _save_results(batch, vectors)

    return EmbeddingRunReport(len(candidates), len(vectors), skipped, failed)
