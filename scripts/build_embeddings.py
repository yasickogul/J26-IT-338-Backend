#!/usr/bin/env python3
"""Backfill shared.legal_chunks.embedding for rows where embedding IS NULL.

Usage (from J26-IT-338-Backend/):
  python -m scripts.build_embeddings
  python -m scripts.build_embeddings --batch-size 32 --sanity-check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from uuid import UUID

import numpy as np
from sqlalchemy import select, text, update

from app.core.database import DirectSessionLocal, direct_engine
from app.core.embeddings import expected_embedding_dim, get_embedder
from app.shared.corpus.models import LegalChunk
from app.shared.corpus.schema import ensure_embedding_dimension, ensure_embedding_index

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


async def run(args: argparse.Namespace) -> int:
    if direct_engine is None or DirectSessionLocal is None:
        print("ERROR: DATABASE_URL / DATABASE_URL_DIRECT is not set", flush=True)
        return 1

    dim = expected_embedding_dim()
    print(f"Target embedding dim: {dim}", flush=True)
    mig = await ensure_embedding_dimension(direct_engine, dim)
    if mig["changed"]:
        print(f"Altered embedding column: {mig['from']} → {mig['to']}", flush=True)
    else:
        print(f"Embedding column already {mig['to']}", flush=True)

    print("Loading embedder (first run may download the model)…", flush=True)
    embedder = get_embedder()
    # Warm-up + dimension check
    probe = embedder.embed_query("dimension probe")
    if len(probe) != dim:
        print(
            f"ERROR: model output dim {len(probe)} != EMBEDDING_DIM={dim}",
            flush=True,
        )
        return 1
    print(f"Embedder ready (probe dim={len(probe)})", flush=True)

    async with DirectSessionLocal() as session:
        total = (
            await session.execute(text("SELECT COUNT(*) FROM shared.legal_chunks"))
        ).scalar_one()
        pending_ids = (
            await session.execute(
                select(LegalChunk.id)
                .where(LegalChunk.embedding.is_(None))
                .order_by(LegalChunk.chunk_index, LegalChunk.id)
            )
        ).scalars().all()

    pending = len(pending_ids)
    already = total - pending
    print(f"Chunks: total={total}  already_embedded={already}  pending={pending}", flush=True)

    if pending == 0:
        print("Nothing to embed.", flush=True)
    else:
        t0 = time.perf_counter()
        embedded = 0
        failures = 0
        batch_size = max(1, args.batch_size)

        for start in range(0, pending, batch_size):
            batch_ids: list[UUID] = list(pending_ids[start : start + batch_size])
            async with DirectSessionLocal() as session:
                rows = (
                    await session.execute(
                        select(LegalChunk.id, LegalChunk.text).where(LegalChunk.id.in_(batch_ids))
                    )
                ).all()
                # Preserve batch order for progress readability
                by_id = {r.id: r.text for r in rows}
                ordered = [(i, by_id[i]) for i in batch_ids if i in by_id]
                texts = [t for _, t in ordered]
                ids = [i for i, _ in ordered]

                try:
                    vectors = embedder.embed_documents(texts)
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL batch starting {start}: {exc}", flush=True)
                    failures += len(ids)
                    continue

                if len(vectors) != len(ids):
                    print("  FAIL batch size mismatch", flush=True)
                    failures += len(ids)
                    continue

                for chunk_id, vec in zip(ids, vectors, strict=True):
                    if len(vec) != dim:
                        failures += 1
                        print(f"  FAIL dim {chunk_id}: got {len(vec)}", flush=True)
                        continue
                    await session.execute(
                        update(LegalChunk)
                        .where(LegalChunk.id == chunk_id)
                        .values(embedding=vec)
                    )
                    embedded += 1
                await session.commit()

            done = already + embedded
            print(f"embedded {done}/{total} chunks", flush=True)

        elapsed = time.perf_counter() - t0
        print("\n=== Embedding summary ===", flush=True)
        print(f"  total chunks     : {total}", flush=True)
        print(f"  newly embedded   : {embedded}", flush=True)
        print(f"  already embedded : {already}", flush=True)
        print(f"  failures         : {failures}", flush=True)
        print(f"  elapsed_seconds  : {elapsed:.1f}", flush=True)

    print("Creating HNSW index (if missing)…", flush=True)
    await ensure_embedding_index(direct_engine)
    print("HNSW index ready.", flush=True)

    if args.sanity_check:
        await _sanity_check(embedder)

    return 0


async def _sanity_check(embedder) -> None:
    print("\n=== Sanity check (self-similarity) ===", flush=True)
    async with DirectSessionLocal() as session:
        row = (
            await session.execute(
                select(LegalChunk.id, LegalChunk.text, LegalChunk.embedding)
                .where(LegalChunk.embedding.is_not(None))
                .limit(1)
            )
        ).one_or_none()

    if row is None:
        print("  SKIP: no embedded chunks found", flush=True)
        return

    chunk_id, text_value, stored = row
    # pgvector may return list or numpy-like
    stored_list = list(stored)
    fresh = embedder.embed_query(text_value)
    sim = _cosine(stored_list, fresh)
    print(f"  chunk_id          : {chunk_id}", flush=True)
    print(f"  stored_dim        : {len(stored_list)}", flush=True)
    print(f"  fresh_dim         : {len(fresh)}", flush=True)
    print(f"  cosine_similarity : {sim:.6f}", flush=True)
    if sim < 0.999:
        print("  WARN: expected ~1.0 — check normalize_embeddings / model mismatch", flush=True)
    else:
        print("  OK: similarity ≈ 1.0", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill chunk embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        default=True,
        help="Re-embed one stored chunk and check cosine ≈ 1.0 (default on)",
    )
    parser.add_argument("--no-sanity-check", action="store_false", dest="sanity_check")
    args = parser.parse_args()

    import asyncio

    return asyncio.run(run(args))


if __name__ == "__main__":
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    raise SystemExit(main())
