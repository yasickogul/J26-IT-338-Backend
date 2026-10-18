"""Database-independent async adapter for C3 legal evidence retrieval.

The search callable is deliberately injected.  Its eventual PostgreSQL
implementation must apply the CompNet contract: filter by the supplied
metadata, rank by ascending vector distance then ``chunk_id`` ascending, and
return at most ``top_k`` ``RetrievalResult`` records without changing their
distance/similarity values.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
import inspect
from typing import Protocol

import numpy as np

from .labse_query_embedding import EXPECTED_DIMENSION
from .claim_input_tokens import _validate_texts
from .retrieval_contract import (
    MAX_TOP_K,
    RetrievalResult,
    _validate_filter,
    _validate_top_k,
)


class AsyncQueryEmbedder(Protocol):
    def __call__(self, query: str) -> Awaitable[np.ndarray]:
        """Return a LaBSE query vector without blocking the event loop."""


class AsyncVectorSearch(Protocol):
    def __call__(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        source_id: str | None,
        legal_domain: str | None,
        language: str | None,
    ) -> Awaitable[Sequence[RetrievalResult]]:
        """Search and return ordered CompNet ``RetrievalResult`` records.

        Implementations must apply all non-None filters, rank by distance
        ascending then chunk ID ascending, and return no more than ``top_k``
        records.  They must preserve scores, IDs, text, and metadata.
        """


def _validate_query(query: str) -> str:
    return _validate_texts([query])[0]


def _validate_query_vector(vector: object) -> np.ndarray:
    values = np.asarray(vector)
    if values.shape != (EXPECTED_DIMENSION,) or not np.isfinite(values).all():
        raise RuntimeError(
            "Query embedding is not a finite 768-dimensional vector"
        )
    return values


def _validate_results(results: object) -> list[RetrievalResult]:
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TypeError("vector search must return a sequence of RetrievalResult records")
    values = list(results)
    if not all(isinstance(result, RetrievalResult) for result in values):
        raise TypeError("vector search must return a sequence of RetrievalResult records")
    return values


async def retrieve(
    query: str,
    *,
    embedder: AsyncQueryEmbedder,
    search: AsyncVectorSearch,
    top_k: int = 5,
    source_id: str | None = None,
    legal_domain: str | None = None,
    language: str | None = None,
) -> list[RetrievalResult]:
    """Return nearest C3 legal chunks through injected async dependencies."""
    _validate_top_k(top_k)
    source_id = _validate_filter("source_id", source_id)
    legal_domain = _validate_filter("legal_domain", legal_domain)
    language = _validate_filter("language", language)
    query = _validate_query(query)
    if not callable(embedder):
        raise TypeError("embedder must be an async callable")
    if not callable(search):
        raise TypeError("search must be an async callable")

    embedding_result = embedder(query)
    if not inspect.isawaitable(embedding_result):
        raise TypeError("embedder must return an awaitable vector")
    query_vector = _validate_query_vector(await embedding_result)

    search_result = search(
        query_vector,
        top_k=top_k,
        source_id=source_id,
        legal_domain=legal_domain,
        language=language,
    )
    if not inspect.isawaitable(search_result):
        raise TypeError("search must return an awaitable result sequence")
    return _validate_results(await search_result)


retrieve_evidence = retrieve


__all__ = [
    "AsyncQueryEmbedder", "AsyncVectorSearch", "MAX_TOP_K", "retrieve",
    "retrieve_evidence",
]
