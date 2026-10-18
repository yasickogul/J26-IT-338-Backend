"""C3-local serialization of CompNet RAG results.

This module contains only the response boundary used by the original RAG
service.  It does not execute retrieval, model calls, or persistence.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .rag_contract import RagResult


class RankedEvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    evidence_id: str
    distance: float
    similarity: float


class EvidencePassageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    chunk_id: str
    original_chunk_id: str
    text: str
    metadata: dict[str, Any]
    source: dict[str, Any]
    parent_context: Any | None = None
    paragraph: str | None = None
    roles: list[str]


class EvidenceRelationshipResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relationship: str
    parent_evidence_id: str
    child_evidence_id: str
    source_id: str


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranked_evidence: list[RankedEvidenceResponse]
    context_evidence_ids: list[str]
    passages: list[EvidencePassageResponse]
    relationships: list[EvidenceRelationshipResponse]
    context_resolution_notices: list[dict[str, Any]]


class ValidationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    validator: str | None = None


class _RagResponseBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_origin: Literal["application", "model"]
    explanation: str | None = None
    citations: list[str]
    insufficient_evidence: bool
    evidence: EvidenceResponse
    limitations: list[str]
    validation: ValidationResponse
    model: str | None = None
    usage: dict[str, Any] | None = None
    attempts: int
    finish_reason: str | None = None


class QuestionRagResponse(_RagResponseBase):
    mode: Literal["question"]
    answer: str | None


class ClaimRagResponse(_RagResponseBase):
    mode: Literal["claim"]
    claim_verdict: Literal["True", "False", "Misleading"] | None


RagResponse = QuestionRagResponse | ClaimRagResponse


def _evidence_response(packet: dict[str, Any]) -> EvidenceResponse:
    """Convert the packet's internal ``evidence`` key to API ``passages``."""
    return EvidenceResponse(
        ranked_evidence=packet.get("ranked_evidence", []),
        context_evidence_ids=packet.get("context_evidence_ids", []),
        passages=packet.get("evidence", []),
        relationships=packet.get("relationships", []),
        context_resolution_notices=packet.get("context_resolution_notices", []),
    )


def serialize_rag_result(result: RagResult) -> RagResponse:
    """Serialize one service result with the original response contract."""
    response = result.response
    common = {
        "response_origin": result.response_origin,
        "explanation": response.get("explanation"),
        "citations": response.get("citations", []),
        "insufficient_evidence": response["insufficient_evidence"],
        "evidence": _evidence_response(result.evidence_packet),
        "limitations": result.limitations,
        "validation": result.validation,
        "model": result.model,
        "usage": result.usage,
        "attempts": result.attempts,
        "finish_reason": result.finish_reason,
    }
    if result.mode == "question":
        return QuestionRagResponse(mode="question", answer=response.get("answer"), **common)
    return ClaimRagResponse(mode="claim", claim_verdict=response.get("claim_verdict"), **common)


# Compatibility name used by the original worker and verification scripts.
_response_model = serialize_rag_result


__all__ = [
    "ClaimRagResponse",
    "EvidencePassageResponse",
    "EvidenceRelationshipResponse",
    "EvidenceResponse",
    "QuestionRagResponse",
    "RagResponse",
    "RankedEvidenceResponse",
    "ValidationResponse",
    "serialize_rag_result",
    "_evidence_response",
    "_response_model",
]
