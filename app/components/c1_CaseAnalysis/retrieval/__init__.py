"""C1-owned semantic retrieval over the shared legal corpus.

Does not modify or replace app.shared.retrieval.HybridRetriever.
"""

from app.components.c1_CaseAnalysis.retrieval.query_builder import build_retrieval_query
from app.components.c1_CaseAnalysis.retrieval.semantic import (
    SemanticRetrievalError,
    SemanticRetriever,
)

__all__ = [
    "SemanticRetrievalError",
    "SemanticRetriever",
    "build_retrieval_query",
]
