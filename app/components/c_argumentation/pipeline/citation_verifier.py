"""Verify that claims cite identifiers present in retrieved sources."""

from collections.abc import Iterable, Mapping


def verify_citations(claim: Mapping[str, object], sources: Iterable[Mapping[str, object]]) -> dict[str, object]:
    source_ids = {str(source.get("id")) for source in sources if source.get("id") is not None}
    citations = [str(value) for value in claim.get("cites", [])] if isinstance(claim.get("cites", []), list) else []
    valid = [citation for citation in citations if citation in source_ids]
    return {"valid": len(valid) == len(citations), "citations": citations, "valid_citations": valid}
