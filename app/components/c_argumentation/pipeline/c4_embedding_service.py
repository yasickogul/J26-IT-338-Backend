"""CPU-local C4 embedding adapter for all-MiniLM-L6-v2."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Sequence

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 16


@lru_cache(maxsize=1)
def get_c4_embedding_model() -> Any:
    """Load one CPU-only SentenceTransformer model instance per process."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ID, device="cpu")


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("Embedding model returned a zero vector")
    return [value / norm for value in values]


def embed_texts(
    texts: Sequence[str],
    *,
    model: Any | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Generate CPU embeddings, normalize them, and enforce 384 dimensions."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not texts:
        return []

    encoder = model or get_c4_embedding_model()
    raw_vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = [_normalize(vector) for vector in raw_vectors]
    invalid = [len(vector) for vector in vectors if len(vector) != EMBEDDING_DIM]
    if invalid:
        raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {invalid[0]}")
    return vectors


def embedding_source_hash(text: str, *, model_id: str = MODEL_ID) -> str:
    """Return a deterministic identity for one model/text embedding input."""

    import hashlib

    payload = f"{model_id}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
