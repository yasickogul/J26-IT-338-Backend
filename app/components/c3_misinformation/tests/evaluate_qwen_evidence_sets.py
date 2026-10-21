"""Evaluate existing C3 evidence sets through evidence assembly and Qwen.

This is an evaluation-only command.  It reads C3 in one read-only transaction,
does not submit claims, and does not alter production retrieval or worker code.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.core.database import SessionLocal
from ..agents.qwen_adapter import DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL, LocalQwenAdapter
from ..pipeline.context_expansion_adapter import expand_parent_context
from ..pipeline.evidence_packet import assemble_evidence_packet
from ..pipeline.evidence_packet_adapter import build_evidence_packet
from ..pipeline.labse_query_embedding import MODEL_NAME, embed_text
from ..pipeline.postgres_parent_lookup import lookup_parents
from ..pipeline.postgres_provenance_lookup import lookup_original_chunk_id
from ..pipeline.postgres_vector_search import C3_CORPUS_CONTRACT, corpus_table, search, validate_passage
from ..pipeline.retrieval_contract import RetrievalResult
from ..prompts.rag_prompt import ClaimInput
from ..rag_contract import RagResponseValidationError
from ..rag_service_async import AsyncRagService, ThreadedGeminiProvider
from .evaluate_hybrid_retrieval import DEFAULT_QUERIES, _build_lexical_query, _lexical_search, _prepared_query, _rrf


SEMANTIC_POOL = 50
LEXICAL_POOL = 50
RRF_K = 60
DISPLAY_TOP_K = 5


def _vector_literal(values: Any) -> str:
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


async def _hydrate_rff_rows(session: Any, fused: list[tuple[Any, Any, int | None, int | None]], query_vector: Any) -> list[RetrievalResult]:
    ids = [item.chunk_id for _, item, _, _ in fused[:DISPLAY_TOP_K]]
    if not ids:
        return []
    table = corpus_table(C3_CORPUS_CONTRACT)
    result = await session.execute(text(f"""
        SELECT chunk_id, original_chunk_id, text, metadata, parent_context, paragraph,
               embedding <=> CAST(:query_embedding AS vector) AS distance,
               1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
        FROM {table}
        WHERE embedding_model = :embedding_model AND chunk_id = ANY(:chunk_ids)
    """), {"query_embedding": _vector_literal(query_vector), "embedding_model": MODEL_NAME, "chunk_ids": ids})
    by_id: dict[str, RetrievalResult] = {}
    for row in result.mappings():
        validate_passage(row)
        by_id[row["chunk_id"]] = RetrievalResult(
            chunk_id=row["chunk_id"], original_chunk_id=row["original_chunk_id"],
            text=row["text"], metadata=row["metadata"], parent_context=row["parent_context"],
            paragraph=row["paragraph"], distance=float(row["distance"]), similarity=float(row["similarity"]),
        )
    return [by_id[item_id] for item_id in ids if item_id in by_id]


async def _packet_for_rows(session: Any, query: str, rows: list[RetrievalResult]) -> dict[str, Any]:
    async def retrieve(_query: str, *, top_k: int, source_id: str | None, legal_domain: str | None, language: str | None) -> list[RetrievalResult]:
        del source_id, legal_domain, language
        return rows[:top_k]

    return await build_evidence_packet(
        query, top_k=DISPLAY_TOP_K, retrieve=retrieve,
        expand_parent_context=partial(expand_parent_context, parent_lookup=partial(lookup_parents, session=session, corpus_contract=C3_CORPUS_CONTRACT)),
        original_chunk_lookup=partial(lookup_original_chunk_id, session=session, corpus_contract=C3_CORPUS_CONTRACT),
        max_parent_depth=1, max_parent_passages=10,
    )


def _synthetic_unrelated_packet(query: str) -> dict[str, Any]:
    unrelated = RetrievalResult(
        chunk_id="synthetic-unrelated-1", original_chunk_id="synthetic-unrelated-1",
        text="A patent application must identify the applicant and include an address for service.",
        metadata={"source_id": "synthetic-unrelated", "title": "Synthetic patent passage", "section_label": "Synthetic"},
        parent_context=None, paragraph=None, distance=0.5, similarity=0.5,
    )
    from ..pipeline.context_expansion import ContextExpansion
    return assemble_evidence_packet(query, [unrelated], ContextExpansion((), ()), original_chunk_ids={}, retrieval_settings={"synthetic": True, "top_k": 1})


async def _verify(service: AsyncRagService, claim: str, packet: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if packet is None:
            result = await service.run(ClaimInput(claim), top_k=DISPLAY_TOP_K)
        else:
            async def builder(_query: str, *, top_k: int, source_id: str | None, legal_domain: str | None, language: str | None) -> dict[str, Any]:
                del top_k, source_id, legal_domain, language
                return packet
            local = AsyncRagService(evidence_packet_builder=builder, provider=service._provider)
            result = await local.run(ClaimInput(claim), top_k=1)
        response = dict(result.response)
        return {"status": "passed", "verdict": response.get("claim_verdict"), "insufficient_evidence": response.get("insufficient_evidence"), "explanation": response.get("explanation"), "citations": response.get("citations", []), "validation": result.validation, "validation_errors": [], "elapsed_seconds": time.perf_counter() - started, "model": result.model, "response_origin": result.response_origin}
    except Exception as exc:
        return {"status": "failed", "verdict": None, "insufficient_evidence": None, "explanation": None, "citations": [], "validation": {}, "validation_errors": [f"{type(exc).__name__}: {exc}"], "elapsed_seconds": time.perf_counter() - started}


async def evaluate(output: Path) -> None:
    if SessionLocal is None:
        raise RuntimeError("C3 DATABASE_URL-backed session factory is unavailable")
    adapter = LocalQwenAdapter(base_url=DEFAULT_QWEN_BASE_URL, model=DEFAULT_QWEN_MODEL)
    service = AsyncRagService(evidence_packet_builder=lambda *args, **kwargs: None, provider=ThreadedGeminiProvider(lambda: adapter))
    document: dict[str, Any] = {"evaluation_only": True, "settings": {"semantic_pool": SEMANTIC_POOL, "lexical_pool": LEXICAL_POOL, "rrf_k": RRF_K, "display_top_k": DISPLAY_TOP_K, "model": DEFAULT_QWEN_MODEL}, "queries": [], "synthetic_unrelated_evidence": None}
    async with SessionLocal() as session:
        async with session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            for query_id, claim in DEFAULT_QUERIES:
                prepared = _prepared_query(claim)
                vector = embed_text(prepared)
                semantic = await search(vector, session=session, corpus_contract=C3_CORPUS_CONTRACT, query_model=MODEL_NAME, top_k=SEMANTIC_POOL, source_id=None, legal_domain=None, language=None)
                lexical = await _lexical_search(session, prepared, LEXICAL_POOL, representation="baseline")
                fused = _rrf(semantic, lexical, RRF_K)
                semantic_packet = await _packet_for_rows(session, prepared, semantic[:DISPLAY_TOP_K])
                rrf_rows = await _hydrate_rff_rows(session, fused, vector)
                rrf_packet = await _packet_for_rows(session, prepared, rrf_rows)
                document["queries"].append({"query_id": query_id, "claim": claim, "prepared_query": prepared, "candidate_counts": {"semantic": len(semantic), "lexical": len(lexical), "rrf_union": len(fused)}, "semantic_top_5": await _verify(service, claim, semantic_packet), "baseline_rrf_top_5": await _verify(service, claim, rrf_packet), "evidence_packets": {"semantic_top_5": semantic_packet, "baseline_rrf_top_5": rrf_packet}})
            synthetic_claim = DEFAULT_QUERIES[0][1]
            synthetic = await _verify(service, synthetic_claim, _synthetic_unrelated_packet(synthetic_claim))
            synthetic["expected_behavior"] = "insufficient_evidence=true; verdict is not prescribed"
            synthetic["behavior_check"] = synthetic.get("insufficient_evidence") is True
            document["synthetic_unrelated_evidence"] = synthetic
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "evaluation" / "qwen_evidence_set_results.json")
    asyncio.run(evaluate(parser.parse_args().output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
