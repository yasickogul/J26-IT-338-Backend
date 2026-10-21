from pathlib import Path

import pytest

from app.components.c_argumentation.pipeline import local_qwen_adapter as adapter


def test_resolve_model_path_accepts_explicit_file(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"test")
    assert adapter.resolve_model_path(path) == path.resolve()


def test_missing_model_is_reported(tmp_path):
    with pytest.raises(adapter.QwenAdapterError, match="GGUF not found"):
        adapter.resolve_model_path(tmp_path / "missing.gguf")


def test_model_loader_is_cached(monkeypatch, tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"test")
    calls = []

    monkeypatch.setattr(adapter, "_load_model", lambda *args: calls.append(args) or object())
    adapter.clear_qwen_cache()
    first = adapter.cached_qwen_model(str(path), 2048, 4)
    second = adapter.cached_qwen_model(str(path), 2048, 4)

    assert first is second
    assert len(calls) == 1


def test_generation_uses_chat_completion_and_strips_reasoning(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"test")

    class FakeModel:
        def create_chat_completion(self, **kwargs):
            assert kwargs["messages"][-1]["content"] == "hello"
            return {"choices": [{"message": {"content": "<think>private</think>Answer."}}]}

    monkeypatch_model = lambda *_args: FakeModel()
    adapter.clear_qwen_cache()
    original = adapter._load_model
    adapter._load_model = monkeypatch_model
    try:
        result = adapter.LocalQwenAdapter(path).generate("hello")
    finally:
        adapter._load_model = original
        adapter.clear_qwen_cache()
    assert result.text == "Answer."
