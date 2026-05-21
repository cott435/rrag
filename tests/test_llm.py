"""Tests for the LLMClient provider router."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from research_rag import Conversation, LLMClient, LLMProvider, Tool
from research_rag.llm import _provider_for


def test_provider_for_claude_routes_to_anthropic():
    assert _provider_for("claude-sonnet-4-5") is LLMProvider.ANTHROPIC
    assert _provider_for("claude-haiku-4-5") is LLMProvider.ANTHROPIC


def test_provider_for_qwen_routes_to_ollama():
    assert _provider_for("qwen3:8b") is LLMProvider.OLLAMA
    assert _provider_for("qwen3:1.7b") is LLMProvider.OLLAMA
    assert _provider_for("llama3.2:3b") is LLMProvider.OLLAMA


def test_supports_tools_only_on_anthropic():
    anthropic = LLMClient(model="claude-sonnet-4-5", client=MagicMock())
    ollama_llm = LLMClient(model="qwen3:8b", client=MagicMock())
    assert anthropic.supports_tools() is True
    assert ollama_llm.supports_tools() is False


def test_conversation_with_tools_on_ollama_raises(tmp_path):
    dummy_tool = Tool(
        name="echo",
        schema={"name": "echo", "description": "x", "input_schema": {"type": "object", "properties": {}}},
        handler=lambda **kw: "ok",
    )
    llm = LLMClient(model="qwen3:8b", client=MagicMock())
    with pytest.raises(ValueError, match="tools are not supported"):
        Conversation(llm=llm, tools=[dummy_tool], db_path=tmp_path / "db.sqlite3")


def test_conversation_without_tools_on_ollama_runs(tmp_path):
    fake_ollama = MagicMock()
    fake_ollama.chat = lambda **kw: {"message": {"content": "hi from qwen"}}
    llm = LLMClient(model="qwen3:8b", client=fake_ollama)
    conv = Conversation(llm=llm, tools=[], db_path=tmp_path / "db.sqlite3")
    assert conv.ask("hello") == "hi from qwen"


def test_ollama_complete_returns_message_content():
    fake_ollama = MagicMock()
    fake_ollama.chat = lambda **kw: {"message": {"content": "42"}}
    llm = LLMClient(model="qwen3:8b", client=fake_ollama)
    assert llm.complete("you are a helper", "what is 6*7?", max_tokens=100) == "42"


def test_ollama_chat_with_tools_raises():
    llm = LLMClient(model="qwen3:8b", client=MagicMock())
    with pytest.raises(ValueError, match="tools are not supported"):
        llm.chat(
            messages=[{"role": "user", "content": "x"}],
            system=None,
            tools=[{"name": "anything"}],
            max_tokens=100,
        )


def test_ollama_passes_system_and_options():
    captured: dict = {}

    def chat(**kw):
        captured.update(kw)
        return {"message": {"content": "ok"}}

    fake_ollama = MagicMock()
    fake_ollama.chat = chat
    llm = LLMClient(model="qwen3:8b", client=fake_ollama)
    llm.complete("SYS", "hello", max_tokens=128, temperature=0.3)
    assert captured["model"] == "qwen3:8b"
    assert captured["messages"][0] == {"role": "system", "content": "SYS"}
    assert captured["messages"][1] == {"role": "user", "content": "hello"}
    assert captured["options"]["num_predict"] == 128
    assert captured["options"]["temperature"] == 0.3
