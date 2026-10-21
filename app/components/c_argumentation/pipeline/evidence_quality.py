"""Transparent retrieval-quality heuristics, not legal relevance judgments."""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval_repository import RetrievedPassage


@dataclass(frozen=True, slots=True)
class EvidenceQuality:
    label: str
    result_count: int
    distinct_judgments: int
    max_vector_similarity: float | None
    supports_legal_conclusion: bool
    explanation: str
    heuristic_note: str = (
        "Heuristic retrieval signal only; similarity and RRF scores are not legal confidence "
        "and do not establish the law or prove a claim."
    )


def assess_evidence(passages: list[RetrievedPassage]) -> EvidenceQuality:
    """Classify retrieval coverage without claiming substantive legal validity."""

    result_count = len(passages)
    distinct_judgments = len({passage.judgment_id for passage in passages})
    similarities = [
        passage.similarity_score
        for passage in passages
        if passage.similarity_score is not None
    ]
    max_similarity = max(similarities) if similarities else None

    if not passages:
        return EvidenceQuality(
            label="empty",
            result_count=0,
            distinct_judgments=0,
            max_vector_similarity=None,
            supports_legal_conclusion=False,
            explanation="No stored passages matched the query.",
        )
    if result_count < 2 or distinct_judgments < 2 or (max_similarity is not None and max_similarity < 0.35):
        return EvidenceQuality(
            label="weak",
            result_count=result_count,
            distinct_judgments=distinct_judgments,
            max_vector_similarity=max_similarity,
            supports_legal_conclusion=False,
            explanation="Retrieved coverage is limited or weak; the passages do not support a legal conclusion by themselves.",
        )
    return EvidenceQuality(
        label="retrieval_support_present",
        result_count=result_count,
        distinct_judgments=distinct_judgments,
        max_vector_similarity=max_similarity,
        supports_legal_conclusion=False,
        explanation="Multiple passages were retrieved, but substantive legal relevance still requires review.",
    )
