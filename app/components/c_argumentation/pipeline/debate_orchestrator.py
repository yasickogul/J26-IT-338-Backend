"""Deterministic debate orchestration seam for future LangGraph integration."""

from collections.abc import Mapping, Sequence

from ..agents.defense_agent import generate_defense_turn
from ..agents.plaintiff_agent import generate_plaintiff_turn


def run_debate(case_facts: Mapping[str, object], rounds: int = 3, sources: Sequence[Mapping[str, object]] = ()) -> list[dict[str, object]]:
    """Generate alternating plaintiff and defense turns for ``rounds`` rounds."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    turns: list[dict[str, object]] = []
    for round_number in range(1, rounds + 1):
        turns.append(generate_plaintiff_turn(case_facts, round_number, sources))
        turns.append(generate_defense_turn(case_facts, round_number, sources))
    return turns
