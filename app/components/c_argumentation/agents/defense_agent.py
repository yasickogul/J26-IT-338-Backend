"""Defense-side agent boundary."""

from collections.abc import Mapping, Sequence


def generate_defense_turn(case_facts: Mapping[str, object], round_number: int, sources: Sequence[Mapping[str, object]] = ()) -> dict[str, object]:
    return {"id": f"defense-{round_number}", "agent": "defense", "round": round_number, "claim": f"Response to: {case_facts.get('facts', '')}", "cites": [str(s["id"]) for s in sources if "id" in s]}
