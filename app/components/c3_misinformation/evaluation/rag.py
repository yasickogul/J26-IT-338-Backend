"""Deterministic structural checks for recorded C3 RAG responses."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ..rag_response import RagResponse


_RESPONSE_ADAPTER = TypeAdapter(RagResponse)
_RAG_STATUSES = {"completed", "unavailable", "not_attempted", "failed"}


def evaluate_rag_response(
    record: Any,
    *,
    expected_request_text: str | None = None,
) -> dict[str, Any]:
    """Report response shape and ID integrity, never semantic legal correctness."""
    if not isinstance(record, Mapping):
        return {"status": "invalid", "issues": ["rag record must be an object"]}
    status = record.get("status", "completed")
    if not isinstance(status, str) or status not in _RAG_STATUSES:
        return {"status": "invalid", "issues": ["unsupported RAG status"]}
    if status != "completed":
        return {
            "status": status,
            "issues": [],
            "schema_valid": None,
            "citation_ids_valid": None,
            "request_text_matches_expected": None,
            "semantic_support": "manual_review_required",
        }

    issues: list[str] = []
    request_text = record.get("request_text")
    if not isinstance(request_text, str) or not request_text.strip():
        issues.append("missing_or_invalid_request_text")
    request_matches = None
    if expected_request_text is not None:
        request_matches = request_text == expected_request_text
        if not request_matches:
            issues.append("request_text_does_not_match_expected")

    mode = record.get("mode")
    if not isinstance(mode, str) or mode not in {"question", "claim"}:
        issues.append("invalid_mode")
    retrieval_query = record.get("retrieval_query", request_text)
    if not isinstance(retrieval_query, str) or not retrieval_query.strip():
        issues.append("invalid_retrieval_query")

    packet = record.get("evidence_packet")
    response = record.get("response")
    packet_valid = (
        isinstance(packet, Mapping)
        and isinstance(packet.get("evidence"), list)
        and isinstance(packet.get("ranked_evidence"), list)
        and isinstance(packet.get("context_evidence_ids"), list)
        and isinstance(packet.get("relationships"), list)
    )
    if not packet_valid:
        issues.append("missing_or_invalid_evidence_packet")
    response_valid = isinstance(response, Mapping)
    if not response_valid:
        issues.append("missing_or_invalid_response")

    schema_valid = False
    citations_valid: bool | None = None
    parsed = None
    if response_valid:
        try:
            parsed = _RESPONSE_ADAPTER.validate_python(dict(response), strict=True)
            schema_valid = True
        except ValidationError:
            issues.append("response_schema_invalid")

        if parsed is not None and parsed.mode != mode:
            issues.append("response_mode_mismatch")
            schema_valid = False
        if parsed is not None:
            if "explanation" not in response or (
                response.get("explanation") is not None
                and not response["explanation"].strip()
            ):
                issues.append("missing_or_blank_explanation")

    packet_ids: list[str] = []
    packet_passages: dict[str, Mapping[str, Any]] = {}
    if packet_valid:
        packet_query = packet.get("query")
        if packet_query != retrieval_query:
            issues.append("evidence_packet_query_mismatch")
        for index, passage in enumerate(packet["evidence"]):
            if not isinstance(passage, Mapping):
                issues.append(f"invalid_evidence_passage:{index}")
                continue
            evidence_id = passage.get("evidence_id")
            chunk_id = passage.get("chunk_id")
            if (not isinstance(evidence_id, str) or not evidence_id.strip()
                    or not isinstance(chunk_id, str) or not chunk_id.strip()):
                issues.append(f"missing_evidence_id:{index}")
                continue
            if evidence_id != f"chunk:{chunk_id}":
                issues.append(f"evidence_id_mismatch:{index}")
            packet_ids.append(evidence_id)
            packet_passages[evidence_id] = passage
        if len(set(packet_ids)) != len(packet_ids):
            issues.append("duplicate_evidence_ids")
        packet_rank_order = []
        ranked_ids = []
        for index, ranked in enumerate(packet["ranked_evidence"]):
            if not isinstance(ranked, Mapping):
                issues.append(f"invalid_packet_ranking:{index}")
                continue
            ranked_id = ranked.get("evidence_id")
            distance = ranked.get("distance")
            similarity = ranked.get("similarity")
            try:
                valid_scores = (
                    not isinstance(distance, bool)
                    and not isinstance(similarity, bool)
                    and isfinite(distance)
                    and isfinite(similarity)
                    and 0.0 <= distance <= 2.0
                    and -1.0 <= similarity <= 1.0
                    and isclose(1.0 - distance, similarity, rel_tol=0.0, abs_tol=1e-6)
                )
            except (TypeError, OverflowError):
                valid_scores = False
            if (type(ranked.get("rank")) is not int
                    or ranked["rank"] != index + 1
                    or not isinstance(ranked_id, str)
                    or ranked_id not in set(packet_ids)
                    or not valid_scores):
                issues.append(f"invalid_packet_ranking:{index}")
            if isinstance(ranked_id, str):
                ranked_ids.append(ranked_id)
            if valid_scores and isinstance(ranked_id, str):
                ranked_passage = next(
                    (passage for passage in packet["evidence"]
                     if isinstance(passage, Mapping)
                     and passage.get("evidence_id") == ranked_id),
                    None,
                )
                if ranked_passage is not None:
                    packet_rank_order.append(
                        (distance, ranked_passage.get("chunk_id", ""))
                    )
        if len(set(ranked_ids)) != len(ranked_ids):
            issues.append("duplicate_packet_ranking_ids")
        if packet_rank_order != sorted(packet_rank_order):
            issues.append("packet_ranking_not_distance_ordered")
        if any(
            not isinstance(item, str) or item not in set(packet_ids)
            for item in packet["context_evidence_ids"]
        ):
            issues.append("packet_context_evidence_id_not_found")
        for index, relationship in enumerate(packet["relationships"]):
            if (not isinstance(relationship, Mapping)
                    or not isinstance(relationship.get("parent_evidence_id"), str)
                    or not isinstance(relationship.get("child_evidence_id"), str)
                    or relationship.get("parent_evidence_id") not in set(packet_ids)
                    or relationship.get("child_evidence_id") not in set(packet_ids)):
                issues.append(f"unresolved_packet_relationship:{index}")

    if response_valid and parsed is not None and packet_valid:
        citations = response.get("citations")
        citations_valid = (
            isinstance(citations, list)
            and all(isinstance(item, str) and item in set(packet_ids) for item in citations)
        )
        if not citations_valid:
            issues.append("citation_id_not_in_evidence_packet")
        if isinstance(citations, list) and len(set(citations)) != len(citations):
            issues.append("duplicate_citation_ids")

        serialized_passages = response.get("evidence", {}).get("passages", [])
        serialized_ids = [
            passage.get("evidence_id")
            for passage in serialized_passages
            if isinstance(passage, Mapping)
        ]
        if len(serialized_ids) != len(serialized_passages) or set(serialized_ids) != set(packet_ids):
            issues.append("response_evidence_differs_from_packet")
        if len(set(serialized_ids)) != len(serialized_ids):
            issues.append("duplicate_response_evidence_ids")
        for serialized in serialized_passages:
            if not isinstance(serialized, Mapping):
                continue
            evidence_id = serialized.get("evidence_id")
            source_passage = packet_passages.get(evidence_id)
            if source_passage is None:
                continue
            fields = (
                "chunk_id", "original_chunk_id", "text", "metadata", "source",
                "parent_context", "paragraph", "roles",
            )
            if any(serialized.get(field) != source_passage.get(field) for field in fields):
                issues.append(f"serialized_evidence_changed:{evidence_id}")

        response_evidence = response.get("evidence", {})
        response_ranked = response_evidence.get("ranked_evidence", [])
        packet_ranked_ids = [
            item.get("evidence_id")
            for item in packet["ranked_evidence"]
            if isinstance(item, Mapping)
        ]
        response_ranked_ids = []
        for index, ranked in enumerate(response_ranked):
            if not isinstance(ranked, Mapping):
                issues.append(f"invalid_response_ranking:{index}")
                continue
            evidence_id = ranked.get("evidence_id")
            response_ranked_ids.append(evidence_id)
            distance = ranked.get("distance")
            similarity = ranked.get("similarity")
            try:
                valid_scores = (
                    not isinstance(distance, bool)
                    and not isinstance(similarity, bool)
                    and isfinite(distance)
                    and isfinite(similarity)
                    and 0.0 <= distance <= 2.0
                    and -1.0 <= similarity <= 1.0
                    and isclose(1.0 - distance, similarity, rel_tol=0.0, abs_tol=1e-6)
                )
            except (TypeError, OverflowError):
                valid_scores = False
            if ranked.get("rank") != index + 1 or evidence_id not in set(packet_ids) or not valid_scores:
                issues.append(f"invalid_response_ranking:{index}")
        if response_ranked_ids != packet_ranked_ids:
            issues.append("response_ranking_differs_from_packet")
        context_ids = response_evidence.get("context_evidence_ids", [])
        if any(item not in set(packet_ids) for item in context_ids):
            issues.append("context_evidence_id_not_in_packet")
        for index, relationship in enumerate(response_evidence.get("relationships", [])):
            if (relationship.get("parent_evidence_id") not in set(packet_ids)
                    or relationship.get("child_evidence_id") not in set(packet_ids)):
                issues.append(f"unresolved_evidence_relationship:{index}")

        insufficient = response.get("insufficient_evidence")
        if not packet_ids:
            if (not insufficient or response.get("citations") != []
                    or response.get("response_origin") != "application"):
                issues.append("empty_evidence_must_use_application_abstention")
        if insufficient:
            if response.get("citations") != []:
                issues.append("insufficient_evidence_must_not_cite")
            if mode == "claim" and response.get("claim_verdict") is not None:
                issues.append("insufficient_claim_must_have_null_verdict")
            if mode == "question" and response.get("answer") is not None:
                issues.append("insufficient_question_must_have_null_answer")

    return {
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "schema_valid": schema_valid,
        "citation_ids_valid": citations_valid,
        "evidence_ids": packet_ids,
        "request_text_matches_expected": request_matches,
        "semantic_support": "manual_review_required",
        "claim_verdict_accuracy": "not_evaluated_no_reviewed_labels",
    }
