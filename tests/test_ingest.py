"""Unit tests for the font-aware heading detector in ingest.py."""
from __future__ import annotations

from research_rag.ingest import _format_lines_as_markdown


def test_above_threshold_line_becomes_heading():
    out = _format_lines_as_markdown(
        [(12.0, "Introduction"), (10.0, "Body line.")],
        body_size=10.0,
    )
    assert out == ["## Introduction", "Body line."]


def test_below_threshold_stays_body():
    """0.4pt above body is below the default 0.5pt threshold."""
    out = _format_lines_as_markdown(
        [(10.4, "Slightly bigger"), (10.0, "Body")],
        body_size=10.0,
    )
    assert out == ["Slightly bigger", "Body"]


def test_long_line_is_body_even_at_heading_size():
    long_text = " ".join(f"word{i}" for i in range(30))
    out = _format_lines_as_markdown(
        [(14.0, long_text)],
        body_size=10.0,
    )
    assert out == [long_text]


def test_consecutive_heading_lines_merge():
    """Title fragments and subsequent headings merge until a body line breaks them."""
    out = _format_lines_as_markdown(
        [
            (16.0, "Title Part One"),
            (16.0, "Title Part Two"),
            (10.0, "Author Names"),
            (12.0, "Section 1"),
            (10.0, "body of section 1"),
        ],
        body_size=10.0,
    )
    assert out == [
        "## Title Part One Title Part Two",
        "Author Names",
        "## Section 1",
        "body of section 1",
    ]


def test_single_char_heading_lines_dropped():
    """1-character lines are too noisy to mark as headings."""
    out = _format_lines_as_markdown(
        [(14.0, "x"), (14.0, "Real Heading"), (10.0, "body")],
        body_size=10.0,
    )
    assert out == ["## Real Heading", "body"]


def test_empty_input():
    assert _format_lines_as_markdown([], body_size=10.0) == []


def test_size_offset_can_be_overridden():
    """A tighter offset catches the LoRA-style 0.8pt bumps."""
    out = _format_lines_as_markdown(
        [(10.4, "Subtle Heading"), (10.0, "body")],
        body_size=10.0,
        size_offset=0.3,
    )
    assert out == ["## Subtle Heading", "body"]
