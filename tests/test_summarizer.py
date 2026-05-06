"""Tests for Summarizer JSON parsing + retry."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from research_rag import SummarizationError, Summarizer


def make_client(text_responses):
    it = iter(text_responses)

    def create(**kw):
        msg = MagicMock(); blk = MagicMock()
        blk.type = "text"; blk.text = next(it)
        msg.content = [blk]
        return msg

    c = MagicMock(); c.messages.create = create
    return c


CANNED = {
    "tldr": "TLDR.", "problem": "P.", "method": "M.", "key_results": "KR.",
    "limitations": "L.", "contributions": ["C1"], "keywords": ["k"],
}


def test_summarizer_valid_json_first_try(isolated_paper_cache, make_paper):
    p = make_paper("a", "content")
    r = Summarizer(client=make_client([json.dumps(CANNED)])).summarize(p)
    assert r.tldr == "TLDR."
    assert (isolated_paper_cache["summaries"] / "a.v1.json").exists()


def test_summarizer_in_memory_cache_hit(isolated_paper_cache, make_paper):
    p = make_paper("a", "content")
    Summarizer(client=make_client([json.dumps(CANNED)])).summarize(p)
    # Same paper instance — _summary populated → no new API call expected.
    cached = Summarizer(client=make_client([])).summarize(p)
    assert cached.tldr == "TLDR."


def test_summarizer_disk_cache_hit(isolated_paper_cache, make_paper):
    p1 = make_paper("a", "content")
    Summarizer(client=make_client([json.dumps(CANNED)])).summarize(p1)
    # Fresh Paper instance with same id — should hit disk cache.
    p2 = make_paper("a", "content")
    cached = Summarizer(client=make_client([])).summarize(p2)
    assert cached.tldr == "TLDR."


def test_summarizer_force_recomputes(isolated_paper_cache, make_paper):
    p = make_paper("a", "content")
    Summarizer(client=make_client([json.dumps(CANNED)])).summarize(p)
    updated = {**CANNED, "tldr": "Updated."}
    r = Summarizer(client=make_client([json.dumps(updated)])).summarize(p, force=True)
    assert r.tldr == "Updated."


def test_summarizer_retries_then_succeeds(isolated_paper_cache, make_paper):
    p = make_paper("b", "content")
    r = Summarizer(client=make_client(["nope", json.dumps(CANNED)])).summarize(p)
    assert r.tldr == "TLDR."


def test_summarizer_raises_after_two_failures(isolated_paper_cache, make_paper):
    p = make_paper("c", "content")
    with pytest.raises(SummarizationError):
        Summarizer(client=make_client(["bad", "still bad"])).summarize(p)


def test_summarizer_strips_code_fences(isolated_paper_cache, make_paper):
    p = make_paper("d", "content")
    fenced = "```json\n" + json.dumps(CANNED) + "\n```"
    r = Summarizer(client=make_client([fenced])).summarize(p)
    assert r.tldr == "TLDR."
