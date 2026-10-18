"""Run lightweight citation and evidence checks over an argument graph."""

from collections.abc import Iterable, Mapping

from .argument_validator import validate_arguments
from .citation_verifier import verify_citations
from .evidence_gap_analyzer import find_evidence_gaps


def audit_graph(graph: Mapping[str, object], sources: Iterable[Mapping[str, object]] = ()) -> dict[str, object]:
    nodes = graph.get("nodes", [])
    source_list = list(sources)
    citation_results = [verify_citations(node, source_list) for node in nodes if isinstance(node, Mapping)]
    return {
        "validation_errors": validate_arguments(nodes),
        "evidence_gaps": find_evidence_gaps(nodes),
        "citation_results": citation_results,
    }
