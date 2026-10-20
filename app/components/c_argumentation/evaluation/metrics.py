"""Small, dependency-free evaluation metrics."""

from collections.abc import Iterable, Mapping


def citation_coverage(nodes: Iterable[Mapping[str, object]]) -> float:
    nodes = list(nodes)
    if not nodes:
        return 1.0
    return sum(bool(node.get("cites")) for node in nodes) / len(nodes)
