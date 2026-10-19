"""Strict validation for explicit evaluation labels and recorded predictions."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite
from typing import Any


class EvaluationInputError(ValueError):
    """Evaluation inputs cannot be scored without changing or guessing data."""


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationInputError(f"{label} must be an object")
    return value


def _nonblank(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationInputError(f"{label} must be a nonblank string")
    return value


def _ids(value: Any, label: str, *, nonempty: bool) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise EvaluationInputError(f"{label} must be a {'nonempty ' if nonempty else ''}list")
    values = [_nonblank(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(set(values)) != len(values):
        raise EvaluationInputError(f"{label} contains duplicate IDs")
    return values


def validate_query_dataset(dataset: Any) -> list[dict[str, Any]]:
    """Validate the reviewed query dataset without rewriting its judgments."""
    source = _object(dataset, "query dataset")
    if source.get("schema_version") != "1.0":
        raise EvaluationInputError("query dataset schema_version must be '1.0'")
    queries = source.get("queries")
    if not isinstance(queries, list) or not queries:
        raise EvaluationInputError("query dataset queries must be a nonempty list")

    ids: set[str] = set()
    for index, raw_query in enumerate(queries):
        query = _object(raw_query, f"queries[{index}]")
        query_id = _nonblank(query.get("query_id"), f"queries[{index}].query_id")
        if query_id in ids:
            raise EvaluationInputError(f"duplicate query_id: {query_id}")
        ids.add(query_id)
        _nonblank(query.get("query"), f"{query_id}.query")
        if query.get("language") != "en":
            raise EvaluationInputError(f"{query_id}.language must preserve reviewed English scope")
        if query.get("provenance") != "constructed_evaluation_query":
            raise EvaluationInputError(f"{query_id}.provenance is unsupported")

        targets = _ids(query.get("target_chunk_ids"), f"{query_id}.target_chunk_ids", nonempty=True)
        backgrounds = _ids(
            query.get("background_chunk_ids", []),
            f"{query_id}.background_chunk_ids",
            nonempty=False,
        )
        if set(targets) & set(backgrounds):
            raise EvaluationInputError(f"{query_id} labels a chunk as both target and background")

        evidence = _object(query.get("evidence"), f"{query_id}.evidence")
        evidence_id = _nonblank(evidence.get("chunk_id"), f"{query_id}.evidence.chunk_id")
        if evidence_id not in targets:
            raise EvaluationInputError(f"{query_id}.evidence.chunk_id is not a designated target")
        _nonblank(evidence.get("text"), f"{query_id}.evidence.text")
        _object(evidence.get("metadata"), f"{query_id}.evidence.metadata")

        alternatives = query.get("alternative_evidence_sets", [])
        if not isinstance(alternatives, list):
            raise EvaluationInputError(f"{query_id}.alternative_evidence_sets must be a list")
        for alternative_index, alternative in enumerate(alternatives):
            alternative = _object(
                alternative,
                f"{query_id}.alternative_evidence_sets[{alternative_index}]",
            )
            _nonblank(
                alternative.get("reason"),
                f"{query_id}.alternative_evidence_sets[{alternative_index}].reason",
            )
            _ids(
                alternative.get("required_chunk_ids"),
                f"{query_id}.alternative_evidence_sets[{alternative_index}].required_chunk_ids",
                nonempty=True,
            )

    scope = _object(source.get("scope"), "query dataset scope")
    revision = source.get("revision", {})
    _object(revision, "query dataset revision")
    expected_count = scope.get("query_count")
    if type(expected_count) is not int or expected_count != len(queries):
        raise EvaluationInputError("query dataset scope.query_count does not match queries")
    return list(queries)


def _validate_retrieval_results(raw_results: Any, query_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw_results, list):
        raise EvaluationInputError(f"{query_id}.retrieval_results must be a list")
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_result in enumerate(raw_results):
        label = f"{query_id}.retrieval_results[{index}]"
        item = _object(raw_result, label)
        rank = item.get("rank")
        if type(rank) is not int or rank != index + 1:
            raise EvaluationInputError(
                f"{label}.rank must be the contiguous one-based rank {index + 1}"
            )
        chunk_id = _nonblank(item.get("chunk_id"), f"{label}.chunk_id")
        if chunk_id in seen_ids:
            raise EvaluationInputError(f"{query_id} has duplicate retrieved chunk_id {chunk_id}")
        seen_ids.add(chunk_id)

        distance = item.get("distance")
        similarity = item.get("similarity")
        try:
            valid_distance = (
                not isinstance(distance, bool)
                and isinstance(distance, (int, float))
                and isfinite(distance)
                and 0.0 <= distance <= 2.0
            )
        except OverflowError:
            valid_distance = False
        if not valid_distance:
            raise EvaluationInputError(f"{label}.distance must be finite and within [0, 2]")
        try:
            valid_similarity = (
                not isinstance(similarity, bool)
                and isinstance(similarity, (int, float))
                and isfinite(similarity)
                and -1.0 <= similarity <= 1.0
            )
        except OverflowError:
            valid_similarity = False
        if not valid_similarity:
            raise EvaluationInputError(f"{label}.similarity must be finite and within [-1, 1]")
        if not isclose(similarity, 1.0 - distance, rel_tol=0.0, abs_tol=1e-6):
            raise EvaluationInputError(f"{label}.similarity must equal 1 - distance")
        results.append(dict(item))

    order = [(item["distance"], item["chunk_id"]) for item in results]
    if order != sorted(order):
        raise EvaluationInputError(
            f"{query_id}.retrieval_results must be ordered by distance then chunk_id"
        )
    return results


def validate_predictions(
    predictions_document: Any,
    query_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Validate prediction IDs and retrieval rankings; preserve absent outputs."""
    source = _object(predictions_document, "predictions document")
    records = source.get("predictions")
    if not isinstance(records, list):
        raise EvaluationInputError("predictions must be a list")

    by_id: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(records):
        record = _object(raw_record, f"predictions[{index}]")
        query_id = _nonblank(record.get("query_id"), f"predictions[{index}].query_id")
        if query_id in by_id:
            raise EvaluationInputError(f"duplicate prediction query_id: {query_id}")
        if query_id not in query_ids:
            raise EvaluationInputError(f"unknown prediction query_id: {query_id}")
        normalized = dict(record)
        if "retrieval_results" in record:
            normalized["retrieval_results"] = _validate_retrieval_results(
                record["retrieval_results"], query_id,
            )
        if "rag" in record and not isinstance(record["rag"], Mapping):
            raise EvaluationInputError(f"{query_id}.rag must be an object")
        if "retrieval_results" not in record and "rag" not in record:
            raise EvaluationInputError(
                f"{query_id} prediction must contain retrieval_results or rag"
            )
        by_id[query_id] = normalized
    return by_id
