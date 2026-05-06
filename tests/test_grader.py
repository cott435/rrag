"""Tests for Grader JSON parsing + retry."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from research_rag import GradeResult, Grader, GradingError


def make_client(text_responses, capture=None):
    it = iter(text_responses)

    def create(**kw):
        if capture is not None:
            capture.append(kw)
        msg = MagicMock(); blk = MagicMock()
        blk.type = "text"; blk.text = next(it)
        msg.content = [blk]
        return msg

    c = MagicMock(); c.messages.create = create
    return c


CANNED = {
    "faithfulness": 4, "relevance": 5, "completeness": 3,
    "citation_quality": 4, "calibration": 4, "overall": 4.0,
    "rationale": "ok", "failure_modes": ["incomplete"],
}


def test_grader_valid_first_try():
    r = Grader(client=make_client([json.dumps(CANNED)])).grade(question="q", answer="a")
    assert isinstance(r, GradeResult)
    assert r.faithfulness == 4 and r.overall == 4.0


def test_grader_failure_modes_optional():
    no_fm = {**CANNED}; no_fm.pop("failure_modes")
    r = Grader(client=make_client([json.dumps(no_fm)])).grade(question="q", answer="a")
    assert r.failure_modes == []


def test_grader_strips_code_fences():
    fenced = "```json\n" + json.dumps(CANNED) + "\n```"
    r = Grader(client=make_client([fenced])).grade(question="q", answer="a")
    assert r.overall == 4.0


def test_grader_retries_then_succeeds():
    r = Grader(client=make_client(["bad", json.dumps(CANNED)])).grade(question="q", answer="a")
    assert r.overall == 4.0


def test_grader_raises_after_two_failures():
    with pytest.raises(GradingError):
        Grader(client=make_client(["bad", "still bad"])).grade(question="q", answer="a")


def test_grader_renders_none_inputs_as_not_provided():
    captured = []
    Grader(client=make_client([json.dumps(CANNED)], capture=captured)).grade(
        question="Q", answer="A"
    )
    user_content = captured[0]["messages"][0]["content"]
    assert "(not provided)" in user_content
