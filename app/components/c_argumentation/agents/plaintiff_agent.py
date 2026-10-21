"""Plaintiff-side agent boundary."""

from collections.abc import Mapping, Sequence

from .multi_agent_agents import PlaintiffAgent


def generate_plaintiff_turn(case_facts: Mapping[str, object], round_number: int, sources: Sequence[Mapping[str, object]] = ()) -> dict[str, object]:
    return {"id": f"plaintiff-{round_number}", "agent": "plaintiff", "round": round_number, "claim": str(case_facts.get("facts", "")), "cites": [str(s["id"]) for s in sources if "id" in s]}
