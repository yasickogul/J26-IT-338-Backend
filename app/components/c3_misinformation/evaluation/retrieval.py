"""Retrieval metrics restricted to the labels present in the reviewed pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import EvaluationInputError


METRIC_DEFINITIONS = {
    "target_hit_at_k": (
        "Fraction of queries with a prediction whose ranking contains any "
        "designated target chunk in the first K positions."
    ),
    "target_mrr_at_k": (
        "Mean reciprocal rank of the first designated target, truncated at K; "
        "queries with no target in the first K contribute zero."
    ),
    "labelled_condition_id_coverage_at_k": (
        "Fraction with every labelled chunk ID for at least one target or "
        "alternative condition in the first K results. Alternative IDs are "
        "separate AND-sets and alternatives are OR paths. This does not verify "
        "structural relationships or legal sufficiency."
    ),
}


def _conditions(query: Mapping[str, Any]) -> list[dict[str, Any]]:
    conditions = [
        {
            "condition_id": f"target:{chunk_id}",
            "kind": "designated_target",
            "required_chunk_ids": [chunk_id],
        }
        for chunk_id in query["target_chunk_ids"]
    ]
    conditions.extend(
        {
            "condition_id": f"alternative:{index + 1}",
            "kind": "alternative_evidence_set",
            "required_chunk_ids": list(item["required_chunk_ids"]),
            "reason": item["reason"],
        }
        for index, item in enumerate(query.get("alternative_evidence_sets", []))
    )
    return conditions


def evaluate_retrieval(
    queries: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Compute deterministic bounded metrics without assuming exhaustive labels."""
    if not isinstance(ks, Sequence) or not ks:
        raise EvaluationInputError("ks must be a nonempty sequence of positive integers")
    if any(type(k) is not int or k < 1 for k in ks):
        raise EvaluationInputError("each K must be a positive integer")
    if len(set(ks)) != len(ks):
        raise EvaluationInputError("ks must not contain duplicates")
    ks = tuple(sorted(ks))

    per_query: list[dict[str, Any]] = []
    evaluated = 0
    missing_retrieval_ids: list[str] = []
    accumulators = {
        k: {"target_hits": 0, "reciprocal_rank_sum": 0.0, "condition_id_coverage": 0}
        for k in ks
    }

    for query in queries:
        query_id = query["query_id"]
        prediction = predictions.get(query_id)
        if prediction is None or "retrieval_results" not in prediction:
            missing_retrieval_ids.append(query_id)
            continue
        evaluated += 1
        results = prediction["retrieval_results"]
        ranks = {item["chunk_id"]: item["rank"] for item in results}
        targets = list(query["target_chunk_ids"])
        target_rank = min((ranks[item] for item in targets if item in ranks), default=None)
        conditions = _conditions(query)
        condition_results = []
        for condition in conditions:
            required = condition["required_chunk_ids"]
            missing = [chunk_id for chunk_id in required if chunk_id not in ranks]
            completion_rank = (
                max(ranks[chunk_id] for chunk_id in required)
                if not missing else None
            )
            condition_results.append({
                **condition,
                "missing_chunk_ids": missing,
                "completion_rank": completion_rank,
                "condition_ids_complete": not missing,
            })

        query_result = {
            "query_id": query_id,
            "returned_chunk_ids": [item["chunk_id"] for item in results],
            "designated_target_rank": target_rank,
            "evidence_conditions": condition_results,
            "metrics": {},
        }
        for k in ks:
            target_hit = target_rank is not None and target_rank <= k
            reciprocal_rank = 1.0 / target_rank if target_hit else 0.0
            completed = [
                condition["condition_id"]
                for condition in condition_results
                if condition["completion_rank"] is not None
                and condition["completion_rank"] <= k
            ]
            condition_coverage = bool(completed)
            accumulators[k]["target_hits"] += int(target_hit)
            accumulators[k]["reciprocal_rank_sum"] += reciprocal_rank
            accumulators[k]["condition_id_coverage"] += int(condition_coverage)
            query_result["metrics"][str(k)] = {
                "target_hit": target_hit,
                "target_reciprocal_rank": reciprocal_rank,
                "labelled_condition_id_coverage": condition_coverage,
                "completed_condition_ids": completed,
            }
        per_query.append(query_result)

    metrics: dict[str, Any] = {}
    for k in ks:
        denominator = evaluated
        values = accumulators[k]
        metrics[str(k)] = {
            "denominator": denominator,
            "target_hit_at_k": {
                "hits": values["target_hits"],
                "rate": values["target_hits"] / denominator if denominator else None,
            },
            "target_mrr_at_k": (
                values["reciprocal_rank_sum"] / denominator if denominator else None
            ),
            "labelled_condition_id_coverage_at_k": {
                "queries_with_complete_condition_ids": values["condition_id_coverage"],
                "rate": (
                    values["condition_id_coverage"] / denominator if denominator else None
                ),
            },
        }
    return {
        "query_count": len(queries),
        "prediction_count": len(predictions),
        "evaluated_retrieval_count": evaluated,
        "missing_retrieval_query_ids": missing_retrieval_ids,
        "metrics": metrics,
        "per_query": per_query,
        "definitions": dict(METRIC_DEFINITIONS),
        "unsupported_metrics": [
            "precision_at_k: relevance labels are not exhaustive",
            "recall_at_k: complete relevant-evidence sets are not established",
            "ndcg_at_k: complete graded relevance judgments are unavailable",
            "full_ranking_mrr: source predictions may be truncated top-K rankings",
            "claim_verdict_accuracy: no reviewed claim-verdict labels are supplied",
        ],
    }
