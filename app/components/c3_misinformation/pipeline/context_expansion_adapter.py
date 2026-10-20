"""Async structural parent expansion; no database or provider access."""
from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import Any, Protocol

from .context_expansion import (
    ParentEvidence, UnresolvedParent, ContextExpansion, _validate_bounds,
    _parent_label,
)
from .retrieval_contract import RetrievalResult


class AsyncParentLookup(Protocol):
    async def __call__(
        self, *, source_id: str, section_label: str,
    ) -> Sequence[Mapping[str, Any]]:
        """Return ALL exact same-source, exact section-label matches.

        Match metadata source_id and section_label without trimming or language
        conversion. Return candidates in chunk_id ascending order, without a
        limit or deduplication: multiple matches must remain ambiguous.
        Each mapping supplies chunk_id, text, metadata, parent_context and
        paragraph, including nullable values. These are source result fields,
        not a declaration of shared database columns.

        Implementations must be asynchronous/nonblocking and own any database
        resources. Filtering and ordering replace the original SQL predicates;
        the orchestration trusts this contract, as it trusted SQL result rows.
        Exceptions propagate unchanged. No default lookup is supplied.
        """


async def expand_parent_context(
    ranked_results: Sequence[RetrievalResult],
    *,
    parent_lookup: AsyncParentLookup,
    max_depth: int = 1,
    max_passages: int = 10,
) -> ContextExpansion:
    """Attach structural parents without changing ranked-result scores or order."""
    _validate_bounds(max_depth, max_passages)
    queue: list[tuple[str, dict[str, Any], int]] = [
        (result.chunk_id, result.metadata, 0) for result in ranked_results
    ]
    parents: list[ParentEvidence] = []
    unresolved: list[UnresolvedParent] = []
    seen_parents: set[str] = set()

    while queue and len(parents) < max_passages:
        child_id, child_metadata, depth = queue.pop(0)
        next_depth = depth + 1
        source_id = child_metadata.get("source_id")
        section_label = child_metadata.get("section_label")
        requested_label = _parent_label(section_label)
        if requested_label is None:
            unresolved.append(UnresolvedParent(
                child_chunk_id=child_id,
                source_id=source_id if isinstance(source_id, str) else None,
                requested_parent_section_label=None,
                depth=next_depth,
                reason="no hierarchical parent label",
            ))
            continue
        if not isinstance(source_id, str) or not source_id.strip():
            unresolved.append(UnresolvedParent(
                child_chunk_id=child_id,
                source_id=None,
                requested_parent_section_label=requested_label,
                depth=next_depth,
                reason="missing source_id",
            ))
            continue

        matches = await parent_lookup(
            source_id=source_id, section_label=requested_label,
        )
        if len(matches) == 0:
            unresolved.append(UnresolvedParent(
                child_chunk_id=child_id,
                source_id=source_id,
                requested_parent_section_label=requested_label,
                depth=next_depth,
                reason="no same-source passage has the parent section label",
            ))
            continue
        if len(matches) != 1:
            unresolved.append(UnresolvedParent(
                child_chunk_id=child_id,
                source_id=source_id,
                requested_parent_section_label=requested_label,
                depth=next_depth,
                reason=f"ambiguous parent: {len(matches)} same-source passages matched",
            ))
            continue

        match = matches[0]
        parent_id = match["chunk_id"]
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        parents.append(ParentEvidence(
            parent_chunk_id=parent_id,
            child_chunk_id=child_id,
            source_id=source_id,
            relationship="structural_parent_of",
            depth=next_depth,
            text=match["text"],
            metadata=match["metadata"],
            parent_context=match["parent_context"],
            paragraph=match["paragraph"],
        ))

        if next_depth < max_depth:
            queue.append((parent_id, match["metadata"], next_depth))

    if queue and len(parents) >= max_passages:
        unresolved.append(UnresolvedParent(
            child_chunk_id=queue[0][0],
            source_id=queue[0][1].get("source_id"),
            requested_parent_section_label=queue[0][1].get("section_label"),
            depth=queue[0][2] + 1,
            reason="parent expansion limit reached",
        ))
    return ContextExpansion(tuple(parents), tuple(unresolved))
