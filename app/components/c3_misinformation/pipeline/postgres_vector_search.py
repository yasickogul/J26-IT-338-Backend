"""Allowlisted PostgreSQL vector-search mappings used by C3 retrieval.

Production C3 composition explicitly selects the C3-owned LaBSE/768 contract.
Legacy public and shared mappings remain available to their existing explicit
callers only; there is no automatic fallback between corpus contracts. Selecting
a contract is not proof that its migration or corpus import has been completed.

All three postgres_* dependencies use caller-owned AsyncSessions. Callers must
establish a read-only transaction and consistent corpus view across retrieval,
parent and original-ID reads, and own commit/rollback/close. No engine, session,
transaction, automatic schema discovery, or production wiring is created here.
"""
from collections.abc import Mapping
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .labse_query_embedding import EXPECTED_DIMENSION, MODEL_NAME
from .retrieval_contract import RetrievalResult, _validate_filter, _validate_top_k

COMPNET_CORPUS_CONTRACT = "compnet-migration-001-public-legal-chunks"
SHARED_CORPUS_CONTRACT = "shared-0001-labse"
C3_CORPUS_CONTRACT = "c3-owned-labse-768"


def require_corpus_contract(corpus_contract: str) -> None:
    if corpus_contract not in (
        COMPNET_CORPUS_CONTRACT,
        SHARED_CORPUS_CONTRACT,
        C3_CORPUS_CONTRACT,
    ):
        raise ValueError("Unsupported corpus mapping")


def corpus_table(corpus_contract: str) -> str:
    """Allowlisted schema mapping only; never interpolate caller identifiers."""
    require_corpus_contract(corpus_contract)
    return {
        COMPNET_CORPUS_CONTRACT: "public.legal_chunks",
        SHARED_CORPUS_CONTRACT: "shared.legal_chunks",
        C3_CORPUS_CONTRACT: "c3.legal_chunks",
    }[corpus_contract]


def validate_passage(row: Mapping[str, Any]) -> None:
    """Fail closed on malformed returned evidence, without inventing metadata.

    Missing citation keys remain absent, as in CompNet; deployment verification
    must audit their availability. Parent lookup SQL enforces exact matching.
    """
    for key in ("chunk_id", "text"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"Invalid corpus {key}")
    if not isinstance(row["metadata"], dict):
        raise ValueError("Corpus metadata must be an object")  # noqa: TRY004 - stable contract
    if row["parent_context"] is not None and not isinstance(row["parent_context"], dict):
        raise ValueError("Corpus parent_context must be an object or null")
    if row["paragraph"] is not None and not isinstance(row["paragraph"], str):
        raise ValueError("Corpus paragraph must be text or null")


async def search(
    query_vector: np.ndarray, *, session: AsyncSession, corpus_contract: str,
    query_model: str, top_k: int = 5, source_id: str | None = None,
    legal_domain: str | None = None, language: str | None = None,
) -> list[RetrievalResult]:
    """Bind session, contract and trusted query_model before adapter injection.

    query_model is a required caller attestation, not inferred from dimensions.
    The SQL additionally selects only stored LaBSE vectors. Errors propagate;
    results are not re-ranked, clamped, normalized or converted from RRF scores.
    """
    require_corpus_contract(corpus_contract)
    if query_model != MODEL_NAME:
        raise ValueError("Query embeddings must use sentence-transformers/LaBSE")
    _validate_top_k(top_k)
    filters = {
        name: _validate_filter(name, value) for name, value in (
            ("source_id", source_id), ("legal_domain", legal_domain), ("language", language),
        )
    }
    vector = np.asarray(query_vector)
    if (vector.shape != (EXPECTED_DIMENSION,)
            or vector.dtype.kind not in "fiu"
            or not np.isfinite(vector).all()):
        raise RuntimeError("Query embedding is not a finite 768-dimensional vector")
    if corpus_contract in {SHARED_CORPUS_CONTRACT, C3_CORPUS_CONTRACT} and not np.allclose(
        np.linalg.norm(vector), 1.0, atol=1e-5, rtol=1e-5,
    ):
        raise ValueError("LaBSE query vector must be unit-normalized")
    conditions = ["embedding_model = :embedding_model"]
    parameters = {
        "query_embedding": "[" + ",".join(repr(float(v)) for v in vector) + "]",
        "embedding_model": MODEL_NAME, "top_k": top_k,
    }
    for name, value in filters.items():
        if value is not None:
            conditions.append(f"metadata ->> '{name}' = :{name}")
            parameters[name] = value
    statement = text(f"""
        SELECT chunk_id, original_chunk_id, text, metadata, parent_context,
               paragraph, distance, 1 - distance AS similarity
        FROM (
            SELECT chunk_id, original_chunk_id, text, metadata, parent_context,
                   paragraph,
                   embedding <=> CAST(:query_embedding AS vector) AS distance
            FROM {corpus_table(corpus_contract)}
            WHERE {' AND '.join(conditions)}
        ) AS ranked
        ORDER BY distance ASC, chunk_id ASC
        LIMIT :top_k
    """)
    result = await session.execute(statement, parameters)
    records = []
    for row in result.mappings():
        validate_passage(row)
        if not isinstance(row["original_chunk_id"], str) or not row["original_chunk_id"].strip():
            raise ValueError("Invalid corpus original_chunk_id")
        records.append(RetrievalResult(
            chunk_id=row["chunk_id"], original_chunk_id=row["original_chunk_id"],
            text=row["text"], metadata=row["metadata"], parent_context=row["parent_context"],
            paragraph=row["paragraph"], distance=float(row["distance"]),
            similarity=float(row["similarity"]),
        ))
    return records
