"""Durable claim state, worker ownership, and result storage."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any
from uuid import UUID


FAILURE_CATEGORIES = frozenset({
    "input", "input_too_long", "request_too_large", "retrieval",
    "preparation", "provider", "response_validation", "service",
    "result_storage", "interrupted",
})


@dataclass(frozen=True)
class ClaimedTextClaim:
    claim_id: UUID
    raw_input_text: str
    worker_token: UUID
    cleaned_text: str | None = None


class ClaimStorageError(RuntimeError):
    """A claim state or result write could not be committed."""


class ClaimStateError(ValueError):
    """The requested claim state transition is not available."""


def _require_uuid(claim_id: UUID) -> None:
    if not isinstance(claim_id, UUID):
        raise ValueError("claim_id must be a UUID")


def _result_snapshot(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if not isinstance(payload, Mapping):
        raise ValueError("result payload must be an object")
    try:
        snapshot = json.loads(json.dumps(dict(payload), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("result payload must be JSON serializable") from exc
    required = {
        "mode", "response_origin", "claim_verdict", "explanation", "citations",
        "insufficient_evidence", "evidence", "limitations", "validation",
        "model", "usage", "attempts", "finish_reason",
    }
    if (not required.issubset(snapshot)
            or snapshot["mode"] != "claim"
            or type(snapshot["insufficient_evidence"]) is not bool
            or not isinstance(snapshot["citations"], list)
            or not isinstance(snapshot["evidence"], dict)
            or not isinstance(snapshot["evidence"].get("passages"), list)):
        raise ValueError("result payload is not a complete claim response")
    if snapshot["insufficient_evidence"]:
        if snapshot["claim_verdict"] is not None:
            raise ValueError("insufficient evidence must have a null claim verdict")
        verdict = "insufficient_evidence"
    else:
        verdict = snapshot["claim_verdict"]
        if verdict not in ("True", "False", "Misleading"):
            raise ValueError("supported claim result has no valid verdict")
    if snapshot["explanation"] is not None and not isinstance(snapshot["explanation"], str):
        raise ValueError("explanation must be text or null")
    if snapshot["model"] is not None and not isinstance(snapshot["model"], str):
        raise ValueError("model must be text or null")
    return snapshot, verdict
