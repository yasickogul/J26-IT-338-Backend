"""Auditor-side agent boundary."""

from collections.abc import Mapping

from .multi_agent_agents import NeutralAuditorAgent


def audit_turn(turn: Mapping[str, object], valid_citation: bool = True) -> dict[str, object]:
    result = dict(turn)
    result["strength_score"] = 1.0 if valid_citation else 0.0
    return result
