"""Independent LaBSE query embedding with injectable model inference.

The production loader remains lazy.  Tests and callers may inject a
SentenceTransformer-compatible encoder; no model weights are loaded at import
time.  Retrieval/database concerns are intentionally outside this module.
"""
from __future__ import annotations

from collections.abc import Sequence
from threading import Lock
from typing import Any

import numpy as np

from .claim_input_tokens import EmbeddingInputTooLongError, _validate_texts


MODEL_NAME = "sentence-transformers/LaBSE"
EXPECTED_DIMENSION = 768

_model: Any | None = None
_model_lock = Lock()


def _get_model() -> Any:
    """Load LaBSE once per process, preserving the source model identity."""
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL_NAME, device="cpu")
            dimension_getter = getattr(
                model,
                "get_embedding_dimension",
                model.get_sentence_embedding_dimension,
            )
            dimension = dimension_getter()
            if dimension != EXPECTED_DIMENSION:
                raise RuntimeError(
                    f"{MODEL_NAME}: expected dimension {EXPECTED_DIMENSION}, "
                    f"got {dimension}."
                )
            _model = model
        return _model


def _encoder_dimension(model: Any) -> int | None:
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:
        getter = getattr(model, "get_sentence_embedding_dimension", None)
    if getter is None:
        return None
    return getter()


def _validate_model_dimension(model: Any) -> None:
    dimension = _encoder_dimension(model)
    if dimension is not None and dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"{MODEL_NAME}: expected dimension {EXPECTED_DIMENSION}, got {dimension}."
        )


def _validate_input_token_limits(values: list[str], model: Any) -> list[int]:
    tokenizer = getattr(model, "tokenizer", None)
    limit = getattr(model, "max_seq_length", None)
    if tokenizer is None or type(limit) is not int:
        raise RuntimeError("LaBSE tokenizer or sequence limit is unavailable.")
    token_ids = tokenizer(
        values, truncation=False, add_special_tokens=True, verbose=False,
    )["input_ids"]
    counts = [len(tokens) for tokens in token_ids]
    for index, token_count in enumerate(counts):
        if token_count > limit:
            raise EmbeddingInputTooLongError(
                index=index, token_count=token_count, limit=limit,
            )
    return counts


def validate_input_token_limits(
    texts: Sequence[str], *, encoder: Any | None = None,
) -> list[int]:
    """Validate full LaBSE token counts, including special tokens."""
    values = _validate_texts(texts)
    model = encoder if encoder is not None else _get_model()
    return _validate_input_token_limits(values, model)


def embed_texts(
    texts: Sequence[str], *, encoder: Any | None = None,
) -> np.ndarray:
    """Return unit-normalized LaBSE vectors in input order."""
    values = _validate_texts(texts)
    model = encoder if encoder is not None else _get_model()
    _validate_model_dimension(model)
    _validate_input_token_limits(values, model)
    vectors = np.asarray(model.encode(
        values,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ))
    expected_shape = (len(values), EXPECTED_DIMENSION)
    if vectors.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected embedding shape {vectors.shape}; expected {expected_shape}."
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embedding output contains non-finite values.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Embedding output is not unit-normalized.")
    return vectors


def embed_text(text: str, *, encoder: Any | None = None) -> np.ndarray:
    """Return one normalized LaBSE vector; reject blank or over-limit text."""
    return embed_texts([text], encoder=encoder)[0]


__all__ = [
    "EXPECTED_DIMENSION", "MODEL_NAME", "EmbeddingInputTooLongError",
    "embed_text", "embed_texts", "validate_input_token_limits",
]
