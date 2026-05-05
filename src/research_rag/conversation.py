"""Conversation — multi-turn chat with a tool-use loop, persisted to SQLite."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from anthropic import Anthropic
from anthropic.types import Message

from .config import CONVERSATIONS_DIR, DEFAULT_INFERENCE_MODEL
from .corpus import PaperCorpus
from .tools import Tool

logger = logging.getLogger(__name__)


class Conversation:
    """Multi-turn chat session with a tool-use loop.

    The loop terminates when the model returns stop_reason != "tool_use"
    or when max_tool_iterations is reached. Local tool dispatch goes
    through each Tool's handler; hosted tools (handler=None) are
    executed server-side by Anthropic and pass through transparently.

    Each user-facing turn is appended to the SQLite turns table, and
    the full message log is snapshotted to conversation_state for resume.
    """

    def __init__(
        self,
        client: Anthropic | None = None,
        corpus: PaperCorpus | None = None,
        model: str = DEFAULT_INFERENCE_MODEL,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        conversation_id: str | None = None,
        max_tokens: int = 2000,
        max_tool_iterations: int = 10,
        db_path: Path | None = None,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
    ):
        self._client = client or Anthropic()
        self.corpus = corpus
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_tool_iterations = max_tool_iterations
        self.on_tool_call = on_tool_call

        self.tools: list[Tool] = list(tools or [])
        self._tool_handlers: dict[str, Callable[..., Any]] = {
            t.name: t.handler for t in self.tools if t.handler is not None
        }
        self._tool_schemas: list[dict] = [t.schema for t in self.tools]

        self.conversation_id = conversation_id or _new_conversation_id()
        self.messages: list[dict] = []

        self._db_path = db_path or (CONVERSATIONS_DIR / "conversations.sqlite3")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        _init_db(self._db_path)

        if conversation_id:
            self._load_history()

    def ask(self, user_message: str) -> str:
        """Send a user message; run the tool loop; return final assistant text."""
        _add_user_message(self.messages, user_message)

        response: Message | None = None
        for _ in range(self.max_tool_iterations):
            response = _chat(
                self._client,
                self.messages,
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=self._tool_schemas,
            )
            _add_assistant_message(self.messages, response)

            if response.stop_reason != "tool_use":
                text = _text_from_message(response)
                self._persist_turn(user_message, text)
                return text

            tool_results: list[dict] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                handler = self._tool_handlers.get(block.name)
                if handler is None:
                    # Hosted tool — already executed server-side; nothing to dispatch.
                    continue
                try:
                    raw = handler(**dict(block.input))
                    result = raw if isinstance(raw, str) else str(raw)
                except Exception as e:
                    logger.exception("Tool %s raised", block.name)
                    result = f"Tool {block.name} raised {type(e).__name__}: {e}"
                if self.on_tool_call:
                    try:
                        self.on_tool_call(block.name, dict(block.input), result)
                    except Exception:
                        logger.exception("on_tool_call hook raised")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

            if tool_results:
                _add_user_message(self.messages, tool_results)

        # Hit the iteration cap — surface what we have plus a note.
        text = _text_from_message(response) if response else ""
        note = "\n\n[max_tool_iterations reached without final response]"
        self._persist_turn(user_message, text + note)
        return text + note

    def _persist_turn(self, user_message: str, assistant_text: str) -> None:
        ts = _now()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO turns (conversation_id, ts, role, content) VALUES (?, ?, ?, ?)",
                (self.conversation_id, ts, "user", user_message),
            )
            conn.execute(
                "INSERT INTO turns (conversation_id, ts, role, content) VALUES (?, ?, ?, ?)",
                (self.conversation_id, ts, "assistant", assistant_text),
            )
            conn.execute(
                "REPLACE INTO conversation_state (conversation_id, ts, messages_json) "
                "VALUES (?, ?, ?)",
                (
                    self.conversation_id,
                    ts,
                    json.dumps(self.messages, default=_serialize_block),
                ),
            )

    def _load_history(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT messages_json FROM conversation_state WHERE conversation_id=?",
                (self.conversation_id,),
            ).fetchone()
        if row:
            self.messages = json.loads(row[0])
            logger.info(
                "Resumed conversation %s with %d messages",
                self.conversation_id,
                len(self.messages),
            )
        else:
            logger.warning(
                "No saved state for conversation %s; starting fresh",
                self.conversation_id,
            )


# Helpers — port of using_tools.ipynb pattern, kept private to this module.

def _add_user_message(messages: list[dict], message: Any) -> None:
    content = message.content if isinstance(message, Message) else message
    messages.append({"role": "user", "content": content})


def _add_assistant_message(messages: list[dict], message: Any) -> None:
    content = message.content if isinstance(message, Message) else message
    messages.append({"role": "assistant", "content": content})


def _chat(
    client: Anthropic,
    messages: list[dict],
    *,
    model: str,
    max_tokens: int,
    system: str | None = None,
    tools: list[dict] | None = None,
    temperature: float = 1.0,
) -> Message:
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        params["system"] = system
    if tools:
        params["tools"] = tools
    return client.messages.create(**params)


def _text_from_message(message: Message) -> str:
    return "\n".join(
        b.text for b in message.content if getattr(b, "type", None) == "text"
    )


def _serialize_block(obj: Any) -> Any:
    """JSON default for Anthropic SDK content blocks (pydantic models)."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"unserializable: {type(obj)}")


def _new_conversation_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_turns_cid ON turns (conversation_id);

            CREATE TABLE IF NOT EXISTS conversation_state (
                conversation_id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                messages_json TEXT NOT NULL
            );
            """
        )
