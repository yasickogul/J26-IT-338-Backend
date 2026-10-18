"""Normalize structured or free-text case facts for the debate pipeline."""

from collections.abc import Mapping


def parse_case_facts(case_facts: Mapping[str, object] | str) -> dict[str, object]:
    """Return a stable case-facts mapping without modifying caller data."""
    if isinstance(case_facts, str):
        return {"facts": case_facts.strip()}
    return dict(case_facts)
