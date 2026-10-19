"""Validation helpers for argument graph nodes."""

from collections.abc import Iterable, Mapping


def validate_arguments(nodes: Iterable[Mapping[str, object]]) -> list[str]:
    """Return validation errors for nodes missing required debate fields."""
    errors: list[str] = []
    for index, node in enumerate(nodes, start=1):
        if not node.get("claim"):
            errors.append(f"node {index} is missing claim")
        if node.get("agent") not in {"plaintiff", "defense", "auditor"}:
            errors.append(f"node {index} has an invalid agent")
    return errors
