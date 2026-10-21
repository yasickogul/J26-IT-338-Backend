from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app.components.c_argumentation.pipeline import c4_embedding_service as service


class FakeEncoder:
    instances = 0

    def __init__(self, *_args, **_kwargs):
        type(self).instances += 1

    def encode(self, texts, **kwargs):
        assert kwargs["batch_size"] == 16
        assert kwargs["normalize_embeddings"] is True
        return [[3.0, 4.0] + [0.0] * 382 for _ in texts]


def test_embeddings_have_384_dimensions_and_unit_norm():
    vectors = service.embed_texts(["ownership of land"], model=FakeEncoder())

    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert sum(value * value for value in vectors[0]) == pytest.approx(1.0)


def test_embedding_model_is_cached(monkeypatch):
    FakeEncoder.instances = 0
    service.get_c4_embedding_model.cache_clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeEncoder))

    first = service.get_c4_embedding_model()
    second = service.get_c4_embedding_model()

    assert first is second
    assert FakeEncoder.instances == 1
    service.get_c4_embedding_model.cache_clear()


def test_source_hash_includes_model_and_exact_text():
    first = service.embedding_source_hash("same text")
    changed_text = service.embedding_source_hash("same text ")
    changed_model = service.embedding_source_hash("same text", model_id="other-model")

    assert first != changed_text
    assert first != changed_model
