"""Parent-context lookup for an explicitly selected corpus contract."""
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .postgres_vector_search import (
    corpus_table,
    require_corpus_contract,
    validate_passage,
)


async def lookup_parents(
    *, source_id: str, section_label: str, session: AsyncSession, corpus_contract: str,
) -> list[dict[str, Any]]:
    """Bind session/contract for AsyncParentLookup; no transaction ownership.

    Match exact metadata values without trimming. Return ALL candidates in
    textual chunk_id order; expansion owns ambiguity, deduplication and bounds.
    No model filter is added here: original parent lookup has none.
    """
    require_corpus_contract(corpus_contract)
    statement = text(f"""
        SELECT chunk_id, text, metadata, parent_context, paragraph
        FROM {corpus_table(corpus_contract)}
        WHERE metadata ->> 'source_id' = :source_id
          AND metadata ->> 'section_label' = :section_label
        ORDER BY chunk_id ASC
    """)
    result = await session.execute(statement, {
        "source_id": source_id, "section_label": section_label,
    })
    rows = []
    for row in result.mappings().all():
        validate_passage(row)
        if (row["metadata"].get("source_id") != source_id
                or row["metadata"].get("section_label") != section_label):
            raise ValueError("Parent metadata does not match requested source and section")
        rows.append(dict(row))
    return rows
