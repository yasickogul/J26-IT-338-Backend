"""C4 semantic retrieval facade, independent of Qwen3 generation."""

from __future__ import annotations

from typing import Any

from .c4_embedding_service import embed_texts
from .evidence_quality import EvidenceQuality, assess_evidence
from .retrieval_repository import (
    RetrievedPassage,
    search_hybrid_judgment_chunks,
    search_judgment_chunks,
)


class C4RagRetriever:
    """Embed a civil-law question locally and retrieve source passages."""

    def __init__(self, *, model: Any | None = None, session_factory: Any = None) -> None:
        self._model = model
        self._session_factory = session_factory

    async def search(self, question: str, *, top_k: int = 5) -> list[RetrievedPassage]:
        """Backward-compatible vector-only search."""

        return await self.search_vector(question, top_k=top_k)

    async def search_vector(
        self,
        question: str,
        *,
        top_k: int = 5,
        max_per_judgment: int | None = None,
    ) -> list[RetrievedPassage]:
        if not question.strip():
            return []
        vector = embed_texts([question], model=self._model, batch_size=1)[0]
        kwargs: dict[str, Any] = {"top_k": top_k, "max_per_judgment": max_per_judgment}
        if self._session_factory is not None:
            kwargs["session_factory"] = self._session_factory
        return await search_judgment_chunks(vector, **kwargs)

    async def search_hybrid(
        self,
        question: str,
        *,
        top_k: int = 5,
        candidate_limit: int = 50,
        max_per_judgment: int | None = 1,
    ) -> list[RetrievedPassage]:
        if not question.strip():
            return []
        vector = embed_texts([question], model=self._model, batch_size=1)[0]
        kwargs: dict[str, Any] = {
            "top_k": top_k,
            "candidate_limit": candidate_limit,
            "max_per_judgment": max_per_judgment,
        }
        if self._session_factory is not None:
            kwargs["session_factory"] = self._session_factory
        return await search_hybrid_judgment_chunks(question, vector, **kwargs)

    @staticmethod
    def assess_evidence(passages: list[RetrievedPassage]) -> EvidenceQuality:
        return assess_evidence(passages)
