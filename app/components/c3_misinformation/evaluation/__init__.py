"""Offline, deterministic C3 retrieval and recorded-RAG evaluation."""

from .contracts import EvaluationInputError, validate_predictions, validate_query_dataset
from .rag import evaluate_rag_response
from .retrieval import evaluate_retrieval
from .runner import build_report, evaluate_documents

__all__ = [
    "EvaluationInputError",
    "build_report",
    "evaluate_documents",
    "evaluate_rag_response",
    "evaluate_retrieval",
    "validate_predictions",
    "validate_query_dataset",
]