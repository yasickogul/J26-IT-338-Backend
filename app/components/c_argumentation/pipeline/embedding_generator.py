"""Generate normalized embeddings with the backend's shared embedder."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from app.core.embeddings import expected_embedding_dim, get_embedder


def generate_embeddings(
    texts: Sequence[str],
    *,
    embedder: Embeddings | None = None,
) -> list[list[float]]:
    """Embed text chunks and verify the configured vector dimension."""

    if not texts:
        return []
    model = embedder or get_embedder()
    vectors = [list(vector) for vector in model.embed_documents(list(texts))]
    expected = expected_embedding_dim()
    invalid = [len(vector) for vector in vectors if len(vector) != expected]
    if invalid:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected}, got {invalid[0]}"
        )
    return vectors
