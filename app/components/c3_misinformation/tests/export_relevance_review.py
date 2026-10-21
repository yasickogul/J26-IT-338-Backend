"""Export a compact, human-labelled C3 retrieval review file.

The exporter is diagnostic only.  It reads semantic results and the existing
baseline RRF comparison from a read-only transaction, deduplicates chunk IDs,
and leaves every relevance label blank.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.core.database import SessionLocal
from ..pipeline.labse_query_embedding import MODEL_NAME, embed_text
from ..pipeline.postgres_vector_search import C3_CORPUS_CONTRACT, search
from .evaluate_hybrid_retrieval import DEFAULT_QUERIES, _build_lexical_query, _prepared_query, _rrf, _lexical_search


SEMANTIC_CANDIDATE_POOL = 50
LEXICAL_CANDIDATE_POOL = 50
RRF_K = 60


def _source_reference(metadata: dict[str, Any]) -> dict[str, Any]:
    source = metadata.get("source_file") or metadata.get("source_url") or metadata.get("pdf_path")
    page = metadata.get("page_number") or metadata.get("pdf_page")
    if page is None:
        section = str(metadata.get("section_label", ""))
        marker = "PDF page "
        if marker in section:
            page = section.split(marker, 1)[1].split(" ", 1)[0]
    return {"document": source if source else "missing", "page": page if page is not None else "missing", "section": metadata.get("section_label", "missing")}


def _passage(row: Any, *, semantic_rank: int | None, rrf_rank: int | None) -> dict[str, Any]:
    metadata = dict(row.metadata)
    return {
        "chunk_id": row.chunk_id,
        "source_id": metadata.get("source_id", "missing"),
        "source_reference": _source_reference(metadata),
        "semantic_rank": semantic_rank,
        "baseline_rrf_rank": rrf_rank,
        "text": row.text,
        "parent_context": getattr(row, "parent_context", None),
        "paragraph": getattr(row, "paragraph", None),
        "relevance_label": "",
        "review_note": "",
    }


def _merge_ranked_passages(semantic: list[Any], fused: list[tuple[Any, Any, int | None, int | None]]) -> list[dict[str, Any]]:
    """Preserve supplied ranks while deduplicating the semantic/RRF union."""
    by_id: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(semantic[:5], 1):
        by_id[row.chunk_id] = _passage(row, semantic_rank=rank, rrf_rank=None)
    for rank, (_, row, semantic_rank, _) in enumerate(fused[:5], 1):
        item = by_id.setdefault(row.chunk_id, _passage(row, semantic_rank=semantic_rank, rrf_rank=None))
        item["baseline_rrf_rank"] = rank
    return list(by_id.values())


async def export(output: Path) -> None:
    if SessionLocal is None:
        raise RuntimeError("C3 DATABASE_URL-backed session factory is unavailable")
    review: dict[str, Any] = {
        "title": "C3 retrieval relevance review",
        "instructions": "Assign relevance_label and review_note manually. Do not infer relevance from source_id alone.",
        "filters": {"embedding_model": MODEL_NAME, "source_id": None, "legal_domain": None, "language": None},
        "settings": {
            "semantic_candidate_pool": SEMANTIC_CANDIDATE_POOL,
            "lexical_candidate_pool": LEXICAL_CANDIDATE_POOL,
            "rrf_k": RRF_K,
            "semantic_top_k": 5,
            "lexical_representation": "baseline",
        },
        "comparison_notes": ["Production retrieval is unchanged."],
        "queries": [],
    }
    async with SessionLocal() as session:
        async with session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            for query_id, raw_query in DEFAULT_QUERIES:
                prepared = _prepared_query(raw_query)
                semantic = await search(embed_text(prepared), session=session, corpus_contract=C3_CORPUS_CONTRACT, query_model=MODEL_NAME, top_k=SEMANTIC_CANDIDATE_POOL, source_id=None, legal_domain=None, language=None)
                lexical = await _lexical_search(session, prepared, LEXICAL_CANDIDATE_POOL, representation="baseline")
                fused = _rrf(semantic, lexical, RRF_K)
                review["queries"].append({
                    "query_id": query_id,
                    "query": raw_query,
                    "prepared_query": prepared,
                    "lexical_query": _build_lexical_query(prepared),
                    "candidate_counts": {"semantic": len(semantic), "lexical": len(lexical), "rrf_input": len(fused)},
                    "passages": _merge_ranked_passages(semantic, fused),
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(review['queries'])} queries, deduplicated passages)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "evaluation" / "relevance_review.json")
    asyncio.run(export(parser.parse_args().output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
