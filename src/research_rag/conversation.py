"""Conversation — multi-turn chat with a tool-use loop, persisted to SQLite."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import CONVERSATIONS_DIR
from .corpus import PaperCorpus
from .llm import LLMClient
from .tools import Tool

logger = logging.getLogger(__name__)


class Conversation:
    """Multi-turn chat session with a tool-use loop.

    The tool-use loop only runs on backends that support tools — currently
    only the Anthropic path. Constructing a Conversation with a non-Anthropic
    :class:`LLMClient` and non-empty ``tools`` raises ``ValueError``. Without
    tools, both providers work; the loop simply terminates after one model
    call since ``stop_reason`` won't be ``tool_use``.

    Each user-facing turn is appended to the SQLite ``turns`` table, and
    the full message log is snapshotted to ``conversation_state`` for resume.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        corpus: PaperCorpus | None = None,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        conversation_id: str | None = None,
        max_tokens: int = 2000,
        max_tool_iterations: int = 10,
        db_path: Path | None = None,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
    ):
        self.llm = llm or LLMClient(model=_default_inference_model())
        self.corpus = corpus
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_tool_iterations = max_tool_iterations
        self.on_tool_call = on_tool_call

        self.tools: list[Tool] = list(tools or [])
        if self.tools and not self.llm.supports_tools():
            raise ValueError(
                f"tools are not supported with provider={self.llm.provider.value} "
                f"(model={self.llm.model!r}); use a Claude model for tool-use loops."
            )
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
        self.messages.append({"role": "user", "content": user_message})

        last_text = ""
        for _ in range(self.max_tool_iterations):
            resp = self.llm.chat(
                messages=self.messages,
                system=self.system_prompt,
                tools=self._tool_schemas or None,
                max_tokens=self.max_tokens,
            )
            self.messages.append({"role": "assistant", "content": resp.raw_assistant_content})
            last_text = resp.text

            if resp.stop_reason != "tool_use":
                self._persist_turn(user_message, resp.text)
                return resp.text

            tool_results: list[dict] = []
            for tu in resp.tool_uses:
                handler = self._tool_handlers.get(tu.name)
                if handler is None:
                    # Hosted tool — already executed server-side; nothing to dispatch.
                    continue
                try:
                    raw = handler(**tu.input)
                    result = raw if isinstance(raw, str) else str(raw)
                except Exception as e:
                    logger.exception("Tool %s raised", tu.name)
                    result = f"Tool {tu.name} raised {type(e).__name__}: {e}"
                if self.on_tool_call:
                    try:
                        self.on_tool_call(tu.name, dict(tu.input), result)
                    except Exception:
                        logger.exception("on_tool_call hook raised")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result,
                    }
                )

            if tool_results:
                self.messages.append({"role": "user", "content": tool_results})

        # Hit the iteration cap — surface what we have plus a note.
        note = "\n\n[max_tool_iterations reached without final response]"
        self._persist_turn(user_message, last_text + note)
        return last_text + note

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


def _default_inference_model() -> str:
    from .config import DEFAULT_INFERENCE_MODEL

    return DEFAULT_INFERENCE_MODEL


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
