"""Read-only comparison of semantic, candidate rerank, and RRF retrieval."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text
from app.core.database import SessionLocal
from ..pipeline.labse_query_embedding import MODEL_NAME, embed_text
from ..pipeline.multilingual_claim_adapter import MultilingualClaimAdapter
from ..pipeline.postgres_vector_search import C3_CORPUS_CONTRACT, corpus_table, search
from ..pipeline.retrieval_contract import RetrievalResult
from ..pipeline.text_preprocessing import clean_text
from ..prompts.rag_prompt import ClaimInput

DEFAULT_QUERIES = (
    ("employment-gratuity", "Employees in Sri Lanka are entitled to gratuity regardless of how long they have worked for their employer."),
    ("employment-service", "A workman with five completed years of service is entitled to gratuity when employment ends."),
    ("consumer-credit", "A hire-purchase agreement must state the cash price of the goods."),
    ("property-intellectual-property", "The law protects rights in an intellectual property work."),
    ("property-land", "Compensation for acquired land may be subject to a claim for unpaid gratuity."),
)
STOP = {"a", "an", "and", "are", "for", "from", "has", "have", "in", "is", "of", "on", "the", "to", "under", "with"}

@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query: str
    target_chunk_ids: tuple[str, ...] = ()
    background_chunk_ids: tuple[str, ...] = ()
    @property
    def labels_available(self) -> bool:
        return bool(self.target_chunk_ids or self.background_chunk_ids)

@dataclass(frozen=True)
class LexicalRow:
    chunk_id: str
    original_chunk_id: str
    text: str
    metadata: dict[str, Any]
    embedding_text: str
    lexical_score: float
    lexical_rank: int

def _tokens(value: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", value.casefold()) if t not in STOP and len(t) > 2}

def _build_lexical_query(value: str) -> str:
    """Build an OR tsquery from the same normalized informative terms used by rerank."""
    return " | ".join(sorted(_tokens(value)))

def _prepared_query(value: str) -> str:
    cleaned = clean_text(value)
    prepared = MultilingualClaimAdapter(detection_enabled=False).prepare_input(ClaimInput(cleaned))
    assert isinstance(prepared, ClaimInput)
    return prepared.retrieval_query or cleaned

def _candidate_rank(query: str, rows: list[RetrievalResult], weight: float) -> list[RetrievalResult]:
    terms = _tokens(query); scored = []
    for row in rows:
        row_terms = _tokens(" ".join((row.text, str(row.metadata.get("title", "")), str(row.metadata.get("section_label", "")))))
        lexical = len(terms & row_terms) / len(terms) if terms else 0.0
        scored.append(((1 - weight) * row.similarity + weight * lexical, row))
    scored.sort(key=lambda x: (-x[0], x[1].distance, x[1].chunk_id))
    return [row for _, row in scored]

def _load_queries(path: Path | None) -> list[QueryCase]:
    if path is None:
        return [QueryCase(i, q) for i, q in DEFAULT_QUERIES]
    raw = json.loads(path.read_text(encoding="utf-8-sig")); raw = raw.get("queries") if isinstance(raw, dict) else raw
    if not isinstance(raw, list) or not raw: raise ValueError("queries file must contain a nonempty queries list")
    out = []
    for n, item in enumerate(raw, 1):
        if not isinstance(item, dict) or not isinstance(item.get("query"), str): raise ValueError(f"queries[{n}] must contain a query string")
        def ids(name: str) -> tuple[str, ...]:
            value = item.get(name, ()); return tuple(str(v) for v in value) if isinstance(value, list) else ()
        out.append(QueryCase(str(item.get("query_id", n)), item["query"], ids("target_chunk_ids"), ids("background_chunk_ids")))
    return out

async def _lexical_search(session: Any, query: str, top_k: int, source_id: str | None = None, legal_domain: str | None = None, language: str | None = None) -> list[LexicalRow]:
    lexical_query = _build_lexical_query(query)
    if not lexical_query:
        return []
    conditions = ["embedding_model = :embedding_model", "to_tsvector('simple', coalesce(embedding_text, '') || ' ' || coalesce(text, '')) @@ to_tsquery('simple', :lexical_query)"]
    params: dict[str, Any] = {"embedding_model": MODEL_NAME, "lexical_query": lexical_query, "top_k": top_k}
    for name, value in (("source_id", source_id), ("legal_domain", legal_domain), ("language", language)):
        if value is not None: conditions.append(f"metadata ->> '{name}' = :{name}"); params[name] = value
    table = corpus_table(C3_CORPUS_CONTRACT)
    stmt = text(f"SELECT chunk_id, original_chunk_id, text, metadata, embedding_text, ts_rank_cd(to_tsvector('simple', coalesce(embedding_text, '') || ' ' || coalesce(text, '')), to_tsquery('simple', :lexical_query)) AS lexical_score FROM {table} WHERE {' AND '.join(conditions)} ORDER BY lexical_score DESC, chunk_id ASC LIMIT :top_k")
    result = await session.execute(stmt, params)
    return [LexicalRow(r["chunk_id"], r["original_chunk_id"], r["text"], r["metadata"], r["embedding_text"], float(r["lexical_score"]), rank) for rank, r in enumerate(result.mappings(), 1)]

def _rrf(semantic: list[RetrievalResult], lexical: list[LexicalRow], k: int):
    merged: dict[str, tuple[Any, int | None, int | None]] = {}
    for rank, row in enumerate(semantic, 1): merged[row.chunk_id] = (row, rank, None)
    for row in lexical:
        old = merged.get(row.chunk_id); merged[row.chunk_id] = (old[0] if old else row, old[1] if old else None, row.lexical_rank)
    out = []
    for row, srank, lrank in merged.values(): out.append(((1/(k+srank) if srank else 0) + (1/(k+lrank) if lrank else 0), row, srank, lrank))
    return sorted(out, key=lambda x: (-x[0], x[1].chunk_id))

def _excerpt(value: str) -> str: return " ".join(value.split())[:180]

def _page_label(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("page_number")
    if value is not None:
        return str(value)
    match = re.search(r"PDF page\\s+(\\d+)", str(metadata.get("section_label", "")), re.I)
    return match.group(1) if match else None

def _print_rows(label: str, rows: list[Any], top_k: int, case: QueryCase):
    print(label)
    for rank, row in enumerate(rows[:top_k], 1):
        if isinstance(row, tuple):
            score, item, srank, lrank = row
            print(f"  {rank}. chunk={item.chunk_id} source={item.metadata.get('source_id','<missing>')} rrf={score:.6f} semantic_rank={srank or '-'} lexical_rank={lrank or '-'} similarity={getattr(item,'similarity',None) or '-'} lexical_score={getattr(item,'lexical_score',None) or '-'} excerpt={_excerpt(item.text)!r}")
        else:
            print(f"  {rank}. chunk={row.chunk_id} source={row.metadata.get('source_id','<missing>')} similarity={getattr(row,'similarity',None) or '-'} lexical_score={getattr(row,'lexical_score',None) or '-'} excerpt={_excerpt(row.text)!r}")
    if case.labels_available:
        found = {getattr(row[1] if isinstance(row, tuple) else row, "chunk_id", "") for row in rows[:top_k]}
        print(f"  labelled_targets_in_top_{top_k}={sorted(found.intersection(case.target_chunk_ids))}")

async def _adjacent_proposal(session: Any, row: RetrievalResult, bound: int):
    source = row.metadata.get("source_id")
    if not source: print("  adjacent_proposal=unavailable (missing source_id)"); return
    table = corpus_table(C3_CORPUS_CONTRACT)
    stmt = text(f"SELECT chunk_id, chunk_index, metadata, metadata ->> 'page_number' AS page_number FROM {table} WHERE source_id=:source_id AND embedding_model=:embedding_model AND chunk_index BETWEEN (SELECT chunk_index FROM {table} WHERE chunk_id=:chunk_id)-:bound AND (SELECT chunk_index FROM {table} WHERE chunk_id=:chunk_id)+:bound ORDER BY chunk_index, chunk_id LIMIT :limit")
    result = await session.execute(stmt, {"source_id": source, "chunk_id": row.chunk_id, "embedding_model": MODEL_NAME, "bound": bound, "limit": bound*2+1})
    anchor_page = _page_label(row.metadata)
    neighbors = []
    for item in result.mappings():
        page = _page_label(item["metadata"])
        neighbors.append({"chunk_id": item["chunk_id"], "chunk_index": item["chunk_index"], "page": page, "cross_page": bool(anchor_page and page and page != anchor_page)})
    print(f"  adjacent_proposal_only source={source} anchor={row.chunk_id} anchor_page={anchor_page or '<unknown>'} rows={neighbors} (reading order only; cross_page is explicit; not ranked or included in evidence)")

async def evaluate(cases: list[QueryCase], candidate_k: int, top_k: int, lexical_weight: float, rrf_k: int, adjacent_bound: int):
    if SessionLocal is None: raise RuntimeError("C3 DATABASE_URL-backed session factory is unavailable")
    async with SessionLocal() as session:
        async with session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            print(json.dumps({"strategy":"diagnostic_only_semantic_candidate_rerank_and_rrf","candidate_k":candidate_k,"top_k":top_k,"lexical_weight":lexical_weight,"rrf_k":rrf_k,"query_model":MODEL_NAME,"corpus_contract":C3_CORPUS_CONTRACT,"filters":{"embedding_model":MODEL_NAME,"source_id":None,"legal_domain":None,"language":None},"production_unchanged":True}, sort_keys=True))
            for case in cases:
                query = _prepared_query(case.query)
                semantic = await search(embed_text(query), session=session, corpus_contract=C3_CORPUS_CONTRACT, query_model=MODEL_NAME, top_k=candidate_k, source_id=None, legal_domain=None, language=None)
                lexical = await _lexical_search(session, query, candidate_k)
                candidate = _candidate_rank(query, semantic, lexical_weight); fused = _rrf(semantic, lexical, rrf_k)
                print(f"\n[{case.query_id}] prepared_query={query!r} lexical_query={_build_lexical_query(query)!r} labels={'available' if case.labels_available else 'missing'}")
                print(f"lexical_candidate_count={len(lexical)}")
                _print_rows("lexical_top_results:", lexical, top_k, case)
                _print_rows(f"semantic_top_{top_k}:", semantic, top_k, case); _print_rows(f"candidate_only_rerank_top_{top_k}:", candidate, top_k, case); _print_rows(f"rrf_top_{top_k}:", fused, top_k, case)
                if semantic: await _adjacent_proposal(session, semantic[0], adjacent_bound)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--queries-file", type=Path); parser.add_argument("--candidate-k", type=int, default=20); parser.add_argument("--top-k", type=int, default=5); parser.add_argument("--lexical-weight", type=float, default=.25); parser.add_argument("--rrf-k", type=int, default=60); parser.add_argument("--adjacent-bound", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.top_k <= args.candidate_k <= 50: parser.error("require 1 <= --top-k <= --candidate-k <= 50")
    if not 0 <= args.lexical_weight <= 1: parser.error("--lexical-weight must be between 0 and 1")
    if args.rrf_k < 1 or not 0 <= args.adjacent_bound <= 5: parser.error("--rrf-k must be positive and --adjacent-bound between 0 and 5")
    asyncio.run(evaluate(_load_queries(args.queries_file), args.candidate_k, args.top_k, args.lexical_weight, args.rrf_k, args.adjacent_bound)); return 0

if __name__ == "__main__": raise SystemExit(main())
