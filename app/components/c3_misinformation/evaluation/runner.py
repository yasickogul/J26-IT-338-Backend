"""Pure evaluation orchestration and reproducible report construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contracts import validate_predictions, validate_query_dataset
from .rag import evaluate_rag_response
from .retrieval import evaluate_retrieval


def evaluate_documents(
    query_dataset: Any,
    predictions_document: Any,
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    queries = validate_query_dataset(query_dataset)
    query_ids = {query["query_id"] for query in queries}
    predictions = validate_predictions(predictions_document, query_ids)
    retrieval = evaluate_retrieval(queries, predictions, ks=ks)

    rag_per_query = []
    missing_rag_ids = []
    for query in queries:
        prediction = predictions.get(query["query_id"])
        if prediction is None or "rag" not in prediction:
            missing_rag_ids.append(query["query_id"])
            continue
        rag_record = prediction["rag"]
        expected = rag_record.get("expected_request_text")
        result = evaluate_rag_response(rag_record, expected_request_text=expected)
        rag_per_query.append({"query_id": query["query_id"], **result})

    return {
        "schema_version": "1.0",
        "report_type": "offline_c3_rag_evaluation",
        "evaluation_scope": {
            "dataset_name": query_dataset.get("dataset_name"),
            "dataset_version": query_dataset.get("revision", {}).get("dataset_version"),
            "query_count": len(queries),
            "query_order": [query["query_id"] for query in queries],
            "review_status": query_dataset.get("review_status"),
            "legal_review_status": query_dataset.get("legal_review_status"),
        },
        "settings": {"k_values": sorted(ks), "ranking": "distance_ascending_then_chunk_id"},
        "retrieval": retrieval,
        "rag": {
            "prediction_count": sum("rag" in item for item in predictions.values()),
            "missing_query_ids": missing_rag_ids,
            "valid_count": sum(item["status"] == "valid" for item in rag_per_query),
            "invalid_count": sum(item["status"] == "invalid" for item in rag_per_query),
            "unavailable_count": sum(
                item["status"] in {"unavailable", "not_attempted", "failed"}
                for item in rag_per_query
            ),
            "per_query": rag_per_query,
            "semantic_citation_support": "manual_review_required",
            "claim_verdict_accuracy": "not_evaluated_no_reviewed_labels",
        },
    }


def build_report(
    query_dataset: Any,
    predictions_document: Any,
    *,
    query_sha256: str,
    predictions_sha256: str,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    report = evaluate_documents(query_dataset, predictions_document, ks=ks)
    report["inputs"] = {
        "query_dataset_sha256": query_sha256,
        "predictions_sha256": predictions_sha256,
    }
    return report
