"""LLM factory behaviour — no live providers, no paid APIs."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from app.core import llm as llm_module


@pytest.mark.unit
def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "llm_provider", "not-a-provider")
    monkeypatch.setattr(llm_module.settings, "llm_model", "x")
    monkeypatch.setattr(llm_module.settings, "llm_api_key", "")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        llm_module.get_llm()


@pytest.mark.unit
def test_openai_provider_constructs_chat_openai(monkeypatch):
    """Successful construction path with a controlled mock (no network)."""
    fake_chat = MagicMock(name="ChatOpenAI")
    fake_instance = MagicMock(name="chat_instance")
    fake_chat.return_value = fake_instance

    fake_mod = ModuleType("langchain_openai")
    fake_mod.ChatOpenAI = fake_chat
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_mod)
    monkeypatch.setattr(llm_module.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm_module.settings, "llm_model", "gpt-test-mock")
    monkeypatch.setattr(llm_module.settings, "llm_api_key", "sk-test")

    result = llm_module.get_llm(temperature=0.0)
    assert result is fake_instance
    fake_chat.assert_called_once()
    kwargs = fake_chat.call_args.kwargs
    assert kwargs["model"] == "gpt-test-mock"
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["temperature"] == 0.0


@pytest.mark.unit
def test_openai_provider_surfaces_connection_failure(monkeypatch):
    """If the client cannot be constructed, the error propagates (no silent fallback)."""
    fake_chat = MagicMock(side_effect=ConnectionError("LLM endpoint unreachable"))
    fake_mod = ModuleType("langchain_openai")
    fake_mod.ChatOpenAI = fake_chat
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_mod)
    monkeypatch.setattr(llm_module.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm_module.settings, "llm_model", "gpt-test-mock")
    monkeypatch.setattr(llm_module.settings, "llm_api_key", "sk-test")

    with pytest.raises(ConnectionError, match="unreachable"):
        llm_module.get_llm()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mocked_generation_via_get_llm_patch(mock_llm):
    """Pattern future explanation agents will use: inject/patch get_llm, never call cloud APIs."""
    from app.core.llm import get_llm

    chat = get_llm()
    result = await chat.ainvoke("Explain why this passage is relevant.")
    assert result.content == "Mocked legal explanation."
    chat.ainvoke.assert_awaited_once()


@pytest.mark.unit
def test_llm_timeout_handling_not_implemented_yet():
    """Document gap: get_llm() has no timeout/retry wrapper yet (architecture §10)."""
    import inspect

    source = inspect.getsource(llm_module.get_llm)
    assert "timeout" not in source.lower()
    assert "retry" not in source.lower()


@pytest.mark.live_llm
def test_live_llm_placeholder_requires_explicit_opt_in():
    """Opt-in marker only — default pytest and CI exclude live_llm."""
    pytest.skip("No LM Studio smoke-test script is wired yet; use an explicit live harness later.")
