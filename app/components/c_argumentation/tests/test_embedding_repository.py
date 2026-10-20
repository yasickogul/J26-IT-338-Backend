from __future__ import annotations

from uuid import uuid4

from app.components.c_argumentation.pipeline import embedding_repository as repository
from app.components.c_argumentation.pipeline.embedding_repository import EmbeddingCandidate


async def test_incremental_run_reports_skips_and_embeds(monkeypatch):
    candidate = EmbeddingCandidate(uuid4(), "legal passage", "hash")
    calls = []

    async def pending(limit):
        assert limit == 1
        return [candidate], 4

    async def mark_processing(candidates):
        calls.append(("processing", candidates))

    async def save_results(candidates, vectors):
        calls.append(("saved", candidates, vectors))

    monkeypatch.setattr(repository, "_pending_candidates", pending)
    monkeypatch.setattr(repository, "_set_processing", mark_processing)
    monkeypatch.setattr(repository, "_save_results", save_results)
    monkeypatch.setattr(repository, "embed_texts", lambda texts, **_kwargs: [[0.0] * 383 + [1.0]])

    report = await repository.embed_pending_chunks(limit=1, batch_size=16, model=object())

    assert report.attempted == 1
    assert report.embedded == 1
    assert report.skipped == 4
    assert report.failed == 0
    assert [call[0] for call in calls] == ["processing", "saved"]


async def test_incremental_run_is_resumable_when_no_candidates(monkeypatch):
    async def pending(limit):
        return [], 7

    monkeypatch.setattr(repository, "_pending_candidates", pending)

    report = await repository.embed_pending_chunks(limit=10)

    assert report.attempted == 0
    assert report.embedded == 0
    assert report.skipped == 7
    assert report.failed == 0
