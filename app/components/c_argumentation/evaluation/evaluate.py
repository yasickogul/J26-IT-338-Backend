"""Evaluate an argument graph with the component's baseline metrics."""

from collections.abc import Mapping

from .metrics import citation_coverage


def evaluate_graph(graph: Mapping[str, object]) -> dict[str, float]:
    nodes = graph.get("nodes", [])
    return {"citation_coverage": citation_coverage(nodes)}
