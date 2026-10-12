"""Shared embedding factory — every component must use this same model."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.core.config import settings

# Settled Week-0 choice (BACKEND_ARCHITECTURE.md §17.1).
# Changing this requires re-embedding the entire corpus.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_embedder() -> Embeddings:
    """Return a LangChain Embeddings object (sentence-transformers via HuggingFace).

    Vectors are L2-normalized so cosine similarity ≡ dot product in pgvector.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = settings.embedding_model or DEFAULT_EMBEDDING_MODEL
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": _pick_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


def expected_embedding_dim() -> int:
    return settings.embedding_dim or DEFAULT_EMBEDDING_DIM


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"
