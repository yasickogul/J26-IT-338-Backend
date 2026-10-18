"""Retrieval adapter used by C4; a shared retriever can be injected later."""

from collections.abc import Iterable, Mapping


def retrieve_sources(case_facts: Mapping[str, object], sources: Iterable[Mapping[str, object]] = ()) -> list[dict[str, object]]:
    """Return supplied source records, preserving the shared retrieval boundary."""
    del case_facts
    return [dict(source) for source in sources]
