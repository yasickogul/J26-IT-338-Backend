"""Resolve structural parent passages for already-ranked retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_PARENT_DEPTH = 3
MAX_PARENT_PASSAGES = 10


@dataclass(frozen=True)
class ParentEvidence:
    parent_chunk_id: str
    child_chunk_id: str
    source_id: str
    relationship: str
    depth: int
    text: str
    metadata: dict[str, Any]
    parent_context: dict[str, Any] | None
    paragraph: str | None


@dataclass(frozen=True)
class UnresolvedParent:
    child_chunk_id: str
    source_id: str | None
    requested_parent_section_label: str | None
    depth: int
    reason: str


@dataclass(frozen=True)
class ContextExpansion:
    parents: tuple[ParentEvidence, ...]
    unresolved: tuple[UnresolvedParent, ...]


def _validate_bounds(max_depth: int, max_passages: int) -> None:
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 1 <= max_depth <= MAX_PARENT_DEPTH:
        raise ValueError(f"max_depth must be between 1 and {MAX_PARENT_DEPTH}")
    if isinstance(max_passages, bool) or not isinstance(max_passages, int) or not 1 <= max_passages <= MAX_PARENT_PASSAGES:
        raise ValueError(f"max_passages must be between 1 and {MAX_PARENT_PASSAGES}")


def _parent_label(section_label: str | None) -> str | None:
    if not isinstance(section_label, str) or " > " not in section_label:
        return None
    return " > ".join(section_label.split(" > ")[:-1])
