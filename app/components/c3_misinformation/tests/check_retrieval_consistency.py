"""Read-only consistency checks for C3 semantic and lexical retrieval."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from typing import Any

import numpy as np
from sqlalchemy import text

from app.core.database import SessionLocal

from ..pipeline.labse_query_embedding import MODEL_NAME, embed_text
from ..pipeline.multilingual_claim_adapter import MultilingualClaimAdapter
from ..pipeline.postgres_vector_search import C3_CORPUS_CONTRACT, corpus_table
from ..pipeline.text_preprocessing import clean_text
from ..prompts.rag_prompt import ClaimInput


DEFAULT_CLAIM = (
    "Employees in Sri Lanka are entitled to gratuity regardless of how long "
    "they have worked for their employer."
)
TARGET_CHUNKS = (
    "PRO-001-0277",
    "PRO-001-0295",
    "CON-003-0014",
    "EMP-010-0010",
    "EMP-010-0011",
)
STOPWORDS = {"a", "an", "and", "are", "for", "from", "has", "have", "in", "is", "of", "on", "the", "to", "under", "with"}


def _prepare_query(claim: str) -> str:
    cleaned = clean_text(claim)
    prepared = MultilingualClaimAdapter(detection_enabled=False).prepare_input(ClaimInput(cleaned))
    assert isinstance(prepared, ClaimInput)
    return prepared.retrieval_query or cleaned


def _lexical_query(value: str) -> str:
    terms = {term for term in re.findall(r"[a-z0-9]+", value.casefold()) if term not in STOPWORDS and len(term) > 2}
    return " | ".join(sorted(terms))


def _vector(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(json.loads(value.replace("{", "[").replace("}", "]")), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _terms(value: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9]+", value.casefold()))


async def _run(session: Any, claim: str, top_k: int) -> None:
    query = _prepare_query(claim)
    lexical_query = _lexical_query(query)
    query_vector = embed_text(query)
    table = corpus_table(C3_CORPUS_CONTRACT)
    print(json.dumps({
        "claim": claim,
        "prepared_query": query,
        "lexical_query": lexical_query,
        "embedding_model": MODEL_NAME,
        "corpus_contract": C3_CORPUS_CONTRACT,
        "filters": {"embedding_model": MODEL_NAME, "source_id": None, "legal_domain": None, "language": None},
        "transaction": "READ ONLY",
    }, sort_keys=True))

    vector_literal = "[" + ",".join(repr(float(value)) for value in query_vector) + "]"
    rows = (await session.execute(text(f"""
        SELECT chunk_id, embedding_text, text, embedding_model,
               embedding::text AS stored_embedding,
               embedding <=> CAST(:query_embedding AS vector) AS sql_distance
        FROM {table}
        WHERE embedding_model = :embedding_model
          AND chunk_id = ANY(:chunk_ids)
        ORDER BY chunk_id ASC
    """), {
        "query_embedding": vector_literal,
        "embedding_model": MODEL_NAME,
        "chunk_ids": list(TARGET_CHUNKS),
    })).mappings().all()

    print("\nVector consistency:")
    for row in rows:
        stored = _vector(row["stored_embedding"])
        fresh = embed_text(row["embedding_text"])
        print(json.dumps({
            "chunk": row["chunk_id"],
            "stored_model": row["embedding_model"],
            "stored_fresh_cosine": _cosine(stored, fresh),
            "stored_fresh_max_abs_diff": float(np.max(np.abs(stored - fresh))),
            "query_stored_python_cosine": _cosine(query_vector, stored),
            "query_stored_sql_cosine": 1.0 - float(row["sql_distance"]),
            "python_sql_cosine_delta": _cosine(query_vector, stored) - (1.0 - float(row["sql_distance"])),
        }, sort_keys=True))

    if not lexical_query:
        print("\nLexical consistency: no usable query terms; no lexical SQL executed")
        return

    lexical_rows = (await session.execute(text(f"""
        WITH scored AS (
            SELECT chunk_id,
                   ts_rank_cd(to_tsvector('simple', coalesce(embedding_text, '')), to_tsquery('simple', :lexical_query)) AS embedding_score,
                   ts_rank_cd(to_tsvector('simple', coalesce(text, '')), to_tsquery('simple', :lexical_query)) AS text_score,
                   ts_rank_cd(to_tsvector('simple', coalesce(embedding_text, '') || ' ' || coalesce(text, '')), to_tsquery('simple', :lexical_query)) AS combined_score
            FROM {table}
            WHERE embedding_model = :embedding_model
              AND to_tsvector('simple', coalesce(embedding_text, '') || ' ' || coalesce(text, '')) @@ to_tsquery('simple', :lexical_query)
        ), ordered AS (
            SELECT *,
                   row_number() OVER (ORDER BY combined_score DESC, chunk_id ASC) AS deterministic_position,
                   rank() OVER (ORDER BY combined_score DESC) AS tied_rank,
                   dense_rank() OVER (ORDER BY combined_score DESC) AS dense_rank
            FROM scored
        )
        SELECT ordered.*, (deterministic_position <= :top_k) AS in_top_k
        FROM ordered
        WHERE chunk_id = ANY(:chunk_ids)
        ORDER BY deterministic_position ASC
    """), {
        "lexical_query": lexical_query,
        "embedding_model": MODEL_NAME,
        "chunk_ids": list(TARGET_CHUNKS),
        "top_k": top_k,
    })).mappings().all()

    print(f"\nLexical consistency (top_k={top_k}, order=combined_score DESC, chunk_id ASC):")
    query_terms = set(lexical_query.replace(" | ", " ").split())
    for row in lexical_rows:
        passage = (await session.execute(text(f"SELECT embedding_text, text FROM {table} WHERE embedding_model=:embedding_model AND chunk_id=:chunk_id"), {"embedding_model": MODEL_NAME, "chunk_id": row["chunk_id"]})).mappings().one()
        embedding_counts = _terms(passage["embedding_text"])
        text_counts = _terms(passage["text"])
        print(json.dumps({
            "chunk": row["chunk_id"],
            "embedding_text_score": float(row["embedding_score"]),
            "text_score": float(row["text_score"]),
            "combined_score": float(row["combined_score"]),
            "deterministic_position": int(row["deterministic_position"]),
            "tied_rank": int(row["tied_rank"]),
            "dense_rank": int(row["dense_rank"]),
            "in_top_k": bool(row["in_top_k"]),
            "matched_terms": sorted(query_terms.intersection(embedding_counts.keys() | text_counts.keys())),
            "embedding_text_query_term_hits": sum(embedding_counts[t] for t in query_terms),
            "text_query_term_hits": sum(text_counts[t] for t in query_terms),
            "combined_query_term_hits": sum(embedding_counts[t] + text_counts[t] for t in query_terms),
            "embedding_text_equals_text": passage["embedding_text"] == passage["text"],
        }, sort_keys=True))


async def diagnose(claim: str, top_k: int) -> None:
    if SessionLocal is None:
        raise RuntimeError("C3 DATABASE_URL-backed session factory is unavailable")
    async with SessionLocal() as session:
        async with session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            await _run(session, claim, top_k)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", default=DEFAULT_CLAIM)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 50:
        parser.error("--top-k must be between 1 and 50")
    asyncio.run(diagnose(args.claim, args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
