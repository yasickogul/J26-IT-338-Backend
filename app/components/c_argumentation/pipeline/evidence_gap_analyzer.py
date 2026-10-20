"""Identify claims that have no supporting citation."""

from collections.abc import Iterable, Mapping


def find_evidence_gaps(nodes: Iterable[Mapping[str, object]]) -> list[str]:
    return [str(node.get("id")) for node in nodes if not node.get("cites") and node.get("id") is not None]
