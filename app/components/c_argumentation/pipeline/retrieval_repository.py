"""Vector, full-text, and hybrid retrieval for C4 judgment chunks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import re
from typing import Any
from uuid import UUID

from sqlalchemy import text

from app.core.database import DirectSessionLocal


VECTOR_RETRIEVAL_SQL = text(
    """
    SELECT
        jc.id AS chunk_id,
        jc.judgment_id AS judgment_id,
        jc.chunk_text AS chunk_text,
        jc.start_page AS start_page,
        jc.end_page AS end_page,
        cc.case_number AS case_number,
        cc.date_decided AS judgment_date,
        cc.pdf_url AS pdf_url,
        1 - (jc.embedding <=> CAST(:query_embedding AS vector)) AS similarity_score
    FROM c_argumentation.judgment_chunks AS jc
    JOIN c_argumentation.case_judgments AS cj ON cj.id = jc.judgment_id
    JOIN c_argumentation.court_cases AS cc ON cc.id = cj.case_id
    WHERE cj.extraction_status = 'extracted'
      AND jc.embedding_status = 'ready'
      AND jc.embedding IS NOT NULL
    ORDER BY jc.embedding <=> CAST(:query_embedding AS vector)
    LIMIT :candidate_limit
    """
)

# PostgreSQL's built-in English configuration is deliberately used here. The
# query text remains a bind parameter; no user text is interpolated into SQL.
FULL_TEXT_RETRIEVAL_SQL = text(
    """
    SELECT
        jc.id AS chunk_id,
        jc.judgment_id AS judgment_id,
        jc.chunk_text AS chunk_text,
        jc.start_page AS start_page,
        jc.end_page AS end_page,
        cc.case_number AS case_number,
        cc.date_decided AS judgment_date,
        cc.pdf_url AS pdf_url,
        ts_rank_cd(
            to_tsvector('english', coalesce(jc.chunk_text, '')),
            websearch_to_tsquery('english', :query_text)
        ) AS full_text_score
    FROM c_argumentation.judgment_chunks AS jc
    JOIN c_argumentation.case_judgments AS cj ON cj.id = jc.judgment_id
    JOIN c_argumentation.court_cases AS cc ON cc.id = cj.case_id
    WHERE cj.extraction_status = 'extracted'
      AND jc.embedding_status = 'ready'
      AND jc.embedding IS NOT NULL
      AND to_tsvector('english', coalesce(jc.chunk_text, ''))
          @@ websearch_to_tsquery('english', :query_text)
    ORDER BY full_text_score DESC, jc.id
    LIMIT :candidate_limit
    """
)

# Backward-compatible name for callers and existing tests.
RETRIEVAL_SQL = VECTOR_RETRIEVAL_SQL
DEFAULT_RRF_K = 60
_FTS_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "of", "on", "or", "that", "the", "to", "what",
    "when", "where", "which", "with",
}


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    chunk_id: UUID
    judgment_id: UUID
    chunk_text: str
    start_page: int | None
    end_page: int | None
    case_number: str
    judgment_date: date | None
    pdf_url: str | None
    similarity_score: float | None
    vector_rank: int | None = None
    full_text_rank: int | None = None
    full_text_score: float | None = None
    rrf_score: float | None = None
    retrieval_method: str = "vector"


def _passage_from_row(row: Any, *, method: str) -> RetrievedPassage:
    values = dict(row)
    return RetrievedPassage(
        chunk_id=values["chunk_id"],
        judgment_id=values["judgment_id"],
        chunk_text=values["chunk_text"],
        start_page=values["start_page"],
        end_page=values["end_page"],
        case_number=values["case_number"],
        judgment_date=values["judgment_date"],
        pdf_url=values["pdf_url"],
        similarity_score=values.get("similarity_score"),
        full_text_score=values.get("full_text_score"),
        retrieval_method=method,
    )


def build_full_text_query(query_text: str) -> str:
    """Build a safe OR-style web-search query from question terms."""

    terms = re.findall(r"[A-Za-z0-9]+", query_text)
    return " OR ".join(
        term for term in terms
        if len(term) >= 3 and term.lower() not in _FTS_STOPWORDS
    )


def diversify_passages(
    passages: list[RetrievedPassage],
    *,
    top_k: int,
    max_per_judgment: int | None,
) -> list[RetrievedPassage]:
    """Select ranked passages while optionally capping one judgment's share."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if max_per_judgment is not None and max_per_judgment < 1:
        raise ValueError("max_per_judgment must be at least 1 when provided")
    if max_per_judgment is None:
        return passages[:top_k]

    selected: list[RetrievedPassage] = []
    counts: dict[UUID, int] = {}
    for passage in passages:
        count = counts.get(passage.judgment_id, 0)
        if count >= max_per_judgment:
            continue
        selected.append(passage)
        counts[passage.judgment_id] = count + 1
        if len(selected) >= top_k:
            break
    return selected


async def search_judgment_chunks(
    query_embedding: list[float],
    *,
    top_k: int = 5,
    max_per_judgment: int | None = None,
    candidate_limit: int | None = None,
    session_factory: Any = DirectSessionLocal,
) -> list[RetrievedPassage]:
    """Search ready C4 embeddings using exact cosine distance."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not query_embedding:
        return []
    if session_factory is None:
        raise RuntimeError("DATABASE_URL/DATABASE_URL_DIRECT is not configured")
    limit = candidate_limit or (max(top_k * 5, 50) if max_per_judgment else top_k)
    if limit < top_k:
        raise ValueError("candidate_limit must be at least top_k")

    async with session_factory() as session:
        rows = (await session.execute(
            VECTOR_RETRIEVAL_SQL,
            {"query_embedding": str(query_embedding), "candidate_limit": limit},
        )).mappings().all()
    passages = [
        replace(_passage_from_row(row, method="vector"), vector_rank=rank)
        for rank, row in enumerate(rows, start=1)
    ]
    return diversify_passages(passages, top_k=top_k, max_per_judgment=max_per_judgment)


def _fuse_rankings(
    vector_rows: list[Any],
    full_text_rows: list[Any],
    *,
    rrf_k: int,
) -> list[RetrievedPassage]:
    fused: dict[UUID, RetrievedPassage] = {}
    for rank, row in enumerate(vector_rows, start=1):
        passage = _passage_from_row(row, method="hybrid")
        fused[passage.chunk_id] = replace(
            passage,
            vector_rank=rank,
            rrf_score=1.0 / (rrf_k + rank),
        )
    for rank, row in enumerate(full_text_rows, start=1):
        passage = _passage_from_row(row, method="hybrid")
        existing = fused.get(passage.chunk_id)
        if existing is None:
            fused[passage.chunk_id] = replace(
                passage,
                full_text_rank=rank,
                rrf_score=1.0 / (rrf_k + rank),
            )
        else:
            fused[passage.chunk_id] = replace(
                existing,
                full_text_rank=rank,
                full_text_score=passage.full_text_score,
                rrf_score=(existing.rrf_score or 0.0) + 1.0 / (rrf_k + rank),
            )
    return sorted(
        fused.values(),
        key=lambda passage: (passage.rrf_score or 0.0, passage.similarity_score or -1.0),
        reverse=True,
    )


async def search_hybrid_judgment_chunks(
    query_text: str,
    query_embedding: list[float],
    *,
    top_k: int = 5,
    candidate_limit: int = 50,
    max_per_judgment: int | None = 1,
    rrf_k: int = DEFAULT_RRF_K,
    session_factory: Any = DirectSessionLocal,
) -> list[RetrievedPassage]:
    """Fuse vector and English full-text rankings with reciprocal rank fusion."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if candidate_limit < top_k:
        raise ValueError("candidate_limit must be at least top_k")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    if not query_text.strip() or not query_embedding:
        return []
    if session_factory is None:
        raise RuntimeError("DATABASE_URL/DATABASE_URL_DIRECT is not configured")

    parameters = {
        "query_embedding": str(query_embedding),
        "query_text": build_full_text_query(query_text),
        "candidate_limit": candidate_limit,
    }
    async with session_factory() as session:
        vector_rows, full_text_rows = await _execute_candidates(session, parameters)
    fused = _fuse_rankings(vector_rows, full_text_rows, rrf_k=rrf_k)
    return diversify_passages(fused, top_k=top_k, max_per_judgment=max_per_judgment)


async def _execute_candidates(session: Any, parameters: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    vector_result = await session.execute(VECTOR_RETRIEVAL_SQL, parameters)
    full_text_result = await session.execute(FULL_TEXT_RETRIEVAL_SQL, parameters)
    return vector_result.mappings().all(), full_text_result.mappings().all()
