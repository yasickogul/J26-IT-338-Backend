"""C1 semantic retriever — pgvector cosine search over shared.legal_chunks.

Uses app.core.embeddings.get_embedder() and existing DB sessions.
Does not modify shared.HybridRetriever or the corpus.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.components.c1_CaseAnalysis.config import c1_settings
from app.components.c1_CaseAnalysis.schemas import SimilarJudgmentMatch
from app.core.embeddings import expected_embedding_dim, get_embedder
from app.shared.contracts.retrieval import RetrievedChunk, SourceCitation

_SEMANTIC_SQL = text(
    """
    WITH ranked AS (
      SELECT
        d.id AS document_id,
        d.title AS title,
        d.citation AS citation,
        d.court AS court,
        d.decision_date AS decision_date,
        d.source_url AS source_url,
        d.source_type AS source_type,
        d.case_type AS case_type,
        c.id AS chunk_id,
        c.chunk_index AS chunk_index,
        c.section AS section,
        c.text AS chunk_text,
        (1 - (c.embedding <=> CAST(:embedding AS vector))) AS similarity,
        ROW_NUMBER() OVER (
          PARTITION BY d.id
          ORDER BY c.embedding <=> CAST(:embedding AS vector)
        ) AS rn
      FROM shared.legal_chunks AS c
      INNER JOIN shared.legal_documents AS d ON d.id = c.document_id
      WHERE c.embedding IS NOT NULL
        AND d.source_type = 'judgment'
        AND (
          :filter_civil = false
          OR d.case_type IS NULL
          OR lower(d.case_type) LIKE '%civil%'
        )
    )
    SELECT
      document_id,
      title,
      citation,
      court,
      decision_date,
      source_url,
      source_type,
      case_type,
      chunk_id,
      chunk_index,
      section,
      chunk_text,
      similarity
    FROM ranked
    WHERE rn = 1
    ORDER BY similarity DESC
    LIMIT :top_k
    """
)


class SemanticRetrievalError(RuntimeError):
    """Raised for embedding or database failures during C1 semantic search."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_top_k(top_k: int | None) -> int:
    default = c1_settings.retrieval_top_k_default
    maximum = c1_settings.retrieval_top_k_max
    value = default if top_k is None else int(top_k)
    if value < 1:
        raise ValueError("top_k must be at least 1")
    if value > maximum:
        raise ValueError(f"top_k must be <= {maximum}")
    return value


def _validate_vector(vector: Sequence[float]) -> list[float]:
    expected = expected_embedding_dim()
    values = [float(x) for x in vector]
    if len(values) != expected:
        raise SemanticRetrievalError(
            "EMBEDDING_DIM_MISMATCH",
            f"Query embedding dimension {len(values)} does not match expected {expected}.",
        )
    if not all(math.isfinite(x) for x in values):
        raise SemanticRetrievalError(
            "EMBEDDING_NON_FINITE",
            "Query embedding contains non-finite values.",
        )
    return values


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vector) + "]"


async def embed_query(query: str) -> list[float]:
    cleaned = " ".join((query or "").split()).strip()
    if not cleaned:
        raise SemanticRetrievalError("EMPTY_QUERY", "Retrieval query must not be empty.")

    def _encode() -> list[float]:
        embedder = get_embedder()
        vectors = embedder.embed_documents([cleaned])
        if not vectors:
            raise SemanticRetrievalError("EMBEDDING_EMPTY", "Embedder returned no vectors.")
        return _validate_vector(vectors[0])

    try:
        return await asyncio.to_thread(_encode)
    except SemanticRetrievalError:
        raise
    except Exception as exc:
        raise SemanticRetrievalError(
            "EMBEDDING_FAILED",
            f"Failed to embed retrieval query: {type(exc).__name__}: {exc}",
        ) from exc


class SemanticRetriever:
    """Distinct-judgment semantic search using pgvector cosine distance."""

    async def search(
        self,
        session: AsyncSession,
        query: str,
        *,
        top_k: int | None = None,
        filter_civil: bool = True,
    ) -> list[SimilarJudgmentMatch]:
        k = validate_top_k(top_k)
        vector = await embed_query(query)
        try:
            result = await session.execute(
                _SEMANTIC_SQL,
                {
                    "embedding": _vector_literal(vector),
                    "top_k": k,
                    "filter_civil": filter_civil,
                },
            )
            rows = result.mappings().all()
        except Exception as exc:
            raise SemanticRetrievalError(
                "DATABASE_SEARCH_FAILED",
                f"Semantic search failed: {type(exc).__name__}: {exc}",
            ) from exc

        matches: list[SimilarJudgmentMatch] = []
        for row in rows:
            similarity = float(row["similarity"] or 0.0)
            matches.append(
                SimilarJudgmentMatch(
                    document_id=str(row["document_id"]),
                    title=row["title"] or "Untitled judgment",
                    citation=row["citation"],
                    court=row["court"],
                    decision_date=row["decision_date"].isoformat()
                    if row["decision_date"] is not None
                    else None,
                    source_url=row["source_url"],
                    source_type=row["source_type"] or "judgment",
                    case_type=row["case_type"],
                    chunk_id=str(row["chunk_id"]),
                    chunk_index=int(row["chunk_index"]),
                    section=row["section"],
                    supporting_excerpt=(row["chunk_text"] or "")[:1200],
                    retrieval_similarity=round(similarity, 6),
                    score_label="retrieval_similarity_cosine",
                )
            )
        return matches

    def to_retrieved_chunks(self, matches: list[SimilarJudgmentMatch]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for match in matches:
            chunks.append(
                RetrievedChunk(
                    chunk_id=match.chunk_id,
                    text=match.supporting_excerpt,
                    score=match.retrieval_similarity,
                    citation=SourceCitation(
                        source_id=match.document_id,
                        title=match.title,
                        source_type=match.source_type,
                        citation=match.citation,
                        section=match.section,
                        url=match.source_url,
                    ),
                    metadata={
                        "document_id": match.document_id,
                        "chunk_index": match.chunk_index,
                        "court": match.court,
                        "decision_date": match.decision_date,
                        "case_type": match.case_type,
                        "score_label": match.score_label,
                    },
                )
            )
        return chunks


def matches_from_raw_rows(rows: list[dict[str, Any]]) -> list[SimilarJudgmentMatch]:
    """Helper for unit tests with mocked row dicts."""
    out: list[SimilarJudgmentMatch] = []
    for row in rows:
        out.append(
            SimilarJudgmentMatch(
                document_id=str(row["document_id"]),
                title=row.get("title") or "Untitled judgment",
                citation=row.get("citation"),
                court=row.get("court"),
                decision_date=row.get("decision_date"),
                source_url=row.get("source_url"),
                source_type=row.get("source_type") or "judgment",
                case_type=row.get("case_type"),
                chunk_id=str(row["chunk_id"]),
                chunk_index=int(row.get("chunk_index") or 0),
                section=row.get("section"),
                supporting_excerpt=(row.get("chunk_text") or row.get("supporting_excerpt") or "")[:1200],
                retrieval_similarity=float(row.get("similarity") or row.get("retrieval_similarity") or 0.0),
                score_label="retrieval_similarity_cosine",
            )
        )
    return out
