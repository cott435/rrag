"""Tests for the Conversation tool-use loop and SQLite persistence."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

from research_rag import Conversation, Tool


# ---- mock helpers ----


def text_block(t):
    b = MagicMock(spec=[])
    b.type = "text"; b.text = t
    return b


def tool_block(name, inputs, tid):
    b = MagicMock(spec=[])
    b.type = "tool_use"; b.name = name; b.input = inputs; b.id = tid
    return b


def msg(stop_reason, content_blocks):
    m = MagicMock(spec=[])
    m.stop_reason = stop_reason; m.content = content_blocks
    return m


def make_client(scripted):
    it = iter(scripted)
    c = MagicMock(); c.messages.create = lambda **kw: next(it)
    return c


def echo_tool(record):
    def echo(text):
        record.append(text)
        return f"echoed: {text}"

    return Tool(
        name="echo",
        schema={
            "name": "echo",
            "description": "echo a string",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
        handler=echo,
    )


# ---- tests ----


def test_dispatches_local_tool_then_returns_text(tmp_path):
    record: list[str] = []
    scripted = [
        msg("tool_use", [tool_block("echo", {"text": "hi"}, "u1")]),
        msg("end_turn", [text_block("done")]),
    ]
    conv = Conversation(
        client=make_client(scripted),
        tools=[echo_tool(record)],
        db_path=tmp_path / "db.sqlite3",
    )
    answer = conv.ask("Echo hi")
    assert answer == "done"
    assert record == ["hi"]


def test_persists_user_and_assistant_turn_to_sqlite(tmp_path):
    db = tmp_path / "db.sqlite3"
    conv = Conversation(
        client=make_client([msg("end_turn", [text_block("hello back")])]),
        tools=[],
        db_path=db,
    )
    conv.ask("hello")
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT role, content FROM turns WHERE conversation_id=? ORDER BY id",
            (conv.conversation_id,),
        ).fetchall()
    assert rows == [("user", "hello"), ("assistant", "hello back")]


def test_tool_exception_is_caught_and_fed_back(tmp_path):
    def boom(**kw):
        raise RuntimeError("kaboom")

    boom_tool = Tool(
        name="boom",
        schema={"name": "boom", "description": "x", "input_schema": {"type": "object", "properties": {}}},
        handler=boom,
    )
    scripted = [
        msg("tool_use", [tool_block("boom", {}, "u1")]),
        msg("end_turn", [text_block("recovered")]),
    ]
    conv = Conversation(
        client=make_client(scripted),
        tools=[boom_tool],
        db_path=tmp_path / "db.sqlite3",
    )
    assert conv.ask("try boom") == "recovered"


def test_max_tool_iterations_cap_terminates_with_marker(tmp_path):
    record: list[str] = []
    scripted = [
        msg("tool_use", [tool_block("echo", {"text": str(i)}, f"u{i}")])
        for i in range(20)
    ]
    conv = Conversation(
        client=make_client(scripted),
        tools=[echo_tool(record)],
        db_path=tmp_path / "db.sqlite3",
        max_tool_iterations=3,
    )
    assert "max_tool_iterations" in conv.ask("loop")


def test_resume_restores_messages_from_sqlite(tmp_path):
    db = tmp_path / "db.sqlite3"
    conv = Conversation(
        client=make_client([msg("end_turn", [text_block("first answer")])]),
        tools=[], db_path=db,
    )
    cid = conv.conversation_id
    conv.ask("hi")
    n1 = len(conv.messages)
    conv2 = Conversation(
        client=make_client([msg("end_turn", [text_block("continued")])]),
        tools=[], db_path=db, conversation_id=cid,
    )
    assert len(conv2.messages) == n1
    assert conv2.ask("more") == "continued"


def test_on_tool_call_hook_receives_name_inputs_result(tmp_path):
    record: list[str] = []
    captured: list[tuple] = []
    scripted = [
        msg("tool_use", [tool_block("echo", {"text": "x"}, "u1")]),
        msg("end_turn", [text_block("done")]),
    ]
    conv = Conversation(
        client=make_client(scripted),
        tools=[echo_tool(record)],
        db_path=tmp_path / "db.sqlite3",
        on_tool_call=lambda n, i, r: captured.append((n, i, r)),
    )
    conv.ask("hi")
    assert captured == [("echo", {"text": "x"}, "echoed: x")]


def test_hosted_tool_with_no_handler_is_ignored_locally(tmp_path):
    """Hosted tools (handler=None) are executed server-side; the local
    dispatch loop should not try to invoke them.
    """
    hosted = Tool(name="web_search", schema={"type": "web_search_20250305", "name": "web_search"}, handler=None)
    scripted = [msg("end_turn", [text_block("hi")])]
    conv = Conversation(
        client=make_client(scripted),
        tools=[hosted],
        db_path=tmp_path / "db.sqlite3",
    )
    # No explosion — the handler map should simply not contain web_search.
    assert "web_search" not in conv._tool_handlers
    assert conv.ask("ping") == "hi"
