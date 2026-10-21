"""Read-only C3 retrieval diagnostic for the Sri Lankan gratuity claim.

This follows the worker's text-cleaning, LaBSE, C3 PostgreSQL vector-search,
and evidence-source contract.  It never calls an LLM, submits a claim, or
writes to the database.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import numpy as np
from sqlalchemy import text

from app.core.database import SessionLocal

from ..pipeline.labse_query_embedding import MODEL_NAME, embed_text
from ..pipeline.multilingual_claim_adapter import MultilingualClaimAdapter
from ..pipeline.postgres_vector_search import C3_CORPUS_CONTRACT, search
from ..pipeline.retrieval_contract import RetrievalResult
from ..pipeline.text_preprocessing import clean_text
from ..prompts.rag_prompt import ClaimInput


DEFAULT_CLAIM = (
    "Employees in Sri Lanka are entitled to gratuity regardless of how long "
    "they have worked for their employer."
)
COMPARISON_SOURCE_ID = "EMP-010"


def prepare_retrieval_query(claim: str) -> tuple[str, str]:
    """Use the same cleaned-claim and English retrieval-query boundary."""
    cleaned = clean_text(claim)
    prepared = MultilingualClaimAdapter(
        detection_enabled=False,
    ).prepare_input(ClaimInput(cleaned))
    assert isinstance(prepared, ClaimInput)
    assert prepared.retrieval_query is not None
    return cleaned, prepared.retrieval_query


def _excerpt(value: str, limit: int = 180) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _print_results(label: str, rows: list[RetrievalResult]) -> None:
    print(f"\n{label} ({len(rows)} results)")
    for rank, row in enumerate(rows, 1):
        source_id = row.metadata.get("source_id", "<missing>")
        print(
            f"{rank:02d}. chunk={row.chunk_id} source={source_id} "
            f"similarity={row.similarity:.8f} distance={row.distance:.8f} "
            f"text={_excerpt(row.text)}"
        )


async def _print_embedding_inputs(session: Any, rows: list[RetrievalResult]) -> None:
    ids = [row.chunk_id for row in rows]
    if not ids:
        return
    records = (await session.execute(text(
        "SELECT chunk_id, embedding_text, embedding_model "
        "FROM c3.legal_chunks WHERE chunk_id = ANY(:chunk_ids) "
        "ORDER BY chunk_id"
    ), {"chunk_ids": ids})).mappings().all()
    print("\nExact stored embedding inputs for displayed semantic results")
    for row in records:
        print(
            f"chunk={row['chunk_id']} model={row['embedding_model']} "
            f"embedding_text={_excerpt(row['embedding_text'], 260)}"
        )


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(json.loads(value.replace("{", "[").replace("}", "]")), dtype=np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


async def _sample_vector_checks(session: Any, *, sample_size: int) -> None:
    model_rows = (await session.execute(text(
        "SELECT embedding_model, COUNT(*) AS rows, COUNT(embedding) AS embeddings "
        "FROM c3.legal_chunks GROUP BY embedding_model ORDER BY embedding_model"
    ))).mappings().all()
    print("\nStored embedding models:")
    for row in model_rows:
        print(
            f"model={row['embedding_model']} rows={row['rows']} "
            f"non_null_embeddings={row['embeddings']}"
        )
    rows = (await session.execute(text(
        "SELECT chunk_id, source_id, embedding_model, embedding_text, "
        "embedding::text AS stored_embedding "
        "FROM c3.legal_chunks "
        "WHERE source_id = :source_id "
        "ORDER BY chunk_id ASC LIMIT :sample_size"
    ), {"source_id": COMPARISON_SOURCE_ID, "sample_size": sample_size})).mappings().all()
    print(f"\nStored-vector checks for {COMPARISON_SOURCE_ID}: {len(rows)} sampled rows")
    for row in rows:
        stored = _parse_vector(row["stored_embedding"])
        fresh = embed_text(row["embedding_text"])
        cosine = float(np.dot(stored, fresh)) if stored.shape == fresh.shape else float("nan")
        max_abs = float(np.max(np.abs(stored - fresh))) if stored.shape == fresh.shape else float("nan")
        print(
            f"chunk={row['chunk_id']} source={row['source_id']} "
            f"model={row['embedding_model']} shape={stored.shape} "
            f"cosine={cosine:.8f} max_abs_diff={max_abs:.8f}"
        )


async def diagnose(claim: str, *, sample_size: int) -> None:
    if SessionLocal is None:
        raise RuntimeError("C3 DATABASE_URL-backed session factory is unavailable")
    cleaned, query = prepare_retrieval_query(claim)
    filters = {
        "source_id": None,
        "legal_domain": None,
        "language": None,
        "top_k": 10,
        "corpus_contract": C3_CORPUS_CONTRACT,
        "query_model": MODEL_NAME,
    }
    print(f"prepared_claim={cleaned}")
    print(f"retrieval_query={query}")
    print(f"active_filters={json.dumps(filters, sort_keys=True)}")
    query_vector = embed_text(query)
    print(f"query_embedding_shape={query_vector.shape} norm={np.linalg.norm(query_vector):.8f}")

    async with SessionLocal() as session:
        async with session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            normal = await search(
                query_vector,
                session=session,
                corpus_contract=C3_CORPUS_CONTRACT,
                query_model=MODEL_NAME,
                top_k=10,
                source_id=None,
                legal_domain=None,
                language=None,
            )
            comparison = await search(
                query_vector,
                session=session,
                corpus_contract=C3_CORPUS_CONTRACT,
                query_model=MODEL_NAME,
                top_k=10,
                source_id=COMPARISON_SOURCE_ID,
                legal_domain=None,
                language=None,
            )
            _print_results("Normal production search", normal)
            _print_results(f"EMP-010 comparison search (diagnostic only)", comparison)
            await _print_embedding_inputs(session, normal[:10])
            await _print_embedding_inputs(session, comparison[:4])
            await _sample_vector_checks(session, sample_size=sample_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", default=DEFAULT_CLAIM)
    parser.add_argument("--sample-size", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.sample_size <= 10:
        parser.error("--sample-size must be between 1 and 10")
    asyncio.run(diagnose(args.claim, sample_size=args.sample_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
