"""LLMClient — provider routing for Anthropic Claude and local Ollama models.

The model string drives the routing decision. Anything starting with ``claude-``
goes through the Anthropic SDK; everything else (e.g. ``qwen3:8b``) goes through
the ``ollama`` package. Both paths return a :class:`NormalizedResponse` so
callers can treat them uniformly.

Tool use is only supported on the Anthropic path. Calling :meth:`LLMClient.chat`
with a non-empty ``tools`` list while routed to Ollama raises ``ValueError`` —
``supports_tools()`` exposes the same gate for callers that need to fail early.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import ollama
from anthropic import Anthropic
from anthropic.types import Message

from .config import OLLAMA_HOST


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict


@dataclass
class NormalizedResponse:
    """Common shape returned by both backends.

    ``raw_assistant_content`` is what callers should re-append to their
    ``messages`` list to continue a multi-turn conversation. For Anthropic
    it's the list of pydantic content blocks (preserves tool_use ids);
    for Ollama it's a plain string.
    """

    text: str
    stop_reason: str
    tool_uses: list[ToolUse] = field(default_factory=list)
    raw_assistant_content: Any = None


def _provider_for(model: str) -> LLMProvider:
    if model.startswith("claude-"):
        return LLMProvider.ANTHROPIC
    return LLMProvider.OLLAMA


class LLMClient:
    """Thin adapter over Anthropic / Ollama, selected by model name.

    ``client`` is optional and primarily a testing seam:
      * Anthropic path: pass a mock with ``.messages.create``.
      * Ollama path: pass an object exposing ``.chat(model, messages, options)``
        (the ``ollama`` module itself satisfies this).
    """

    def __init__(
        self,
        model: str,
        client: Any | None = None,
        ollama_host: str | None = None,
    ):
        self.model = model
        self.provider = _provider_for(model)
        if self.provider is LLMProvider.ANTHROPIC:
            self._client = client or Anthropic()
        else:
            if client is not None:
                self._client = client
            else:
                host = ollama_host or OLLAMA_HOST
                self._client = ollama.Client(host=host) if host else ollama

    def supports_tools(self) -> bool:
        return self.provider is LLMProvider.ANTHROPIC

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float | None = None,
    ) -> str:
        """Single-shot completion. Returns the assistant text only."""
        resp = self.chat(
            messages=[{"role": "user", "content": user}],
            system=system,
            tools=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.text

    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int,
        temperature: float | None = None,
    ) -> NormalizedResponse:
        """Send a multi-turn request and return a normalized response.

        For the Ollama path, ``tools`` must be empty/None — Qwen support is
        chat-only in this codebase. The caller is expected to gate on
        :meth:`supports_tools` if mixing providers.
        """
        if self.provider is LLMProvider.ANTHROPIC:
            return self._anthropic_chat(messages, system, tools, max_tokens, temperature)
        return self._ollama_chat(messages, system, tools, max_tokens, temperature)

    def _anthropic_chat(
        self,
        messages: list[dict],
        system: str | None,
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float | None,
    ) -> NormalizedResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature

        response: Message = self._client.messages.create(**kwargs)
        text = "\n".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
        tool_uses = [
            ToolUse(id=b.id, name=b.name, input=dict(b.input))
            for b in response.content
            if getattr(b, "type", None) == "tool_use"
        ]
        return NormalizedResponse(
            text=text,
            stop_reason=response.stop_reason or "end_turn",
            tool_uses=tool_uses,
            raw_assistant_content=response.content,
        )

    def _ollama_chat(
        self,
        messages: list[dict],
        system: str | None,
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float | None,
    ) -> NormalizedResponse:
        if tools:
            raise ValueError(
                "tools are not supported with provider=ollama "
                f"(model={self.model!r}); use a Claude model for tool-use loops."
            )

        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        options: dict[str, Any] = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature

        response = self._client.chat(
            model=self.model,
            messages=msgs,
            options=options,
        )
        # ollama returns a dict-like; .message.content on newer SDKs, ["message"]["content"] always.
        message = response["message"] if isinstance(response, dict) else response.message
        text = message["content"] if isinstance(message, dict) else message.content
        return NormalizedResponse(
            text=text,
            stop_reason="end_turn",
            tool_uses=[],
            raw_assistant_content=text,
        )
