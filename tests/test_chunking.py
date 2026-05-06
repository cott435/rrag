"""Tests for the section-aware chunker."""
from __future__ import annotations

from research_rag import chunk_paper
from research_rag.chunking import chunk_by_section, chunk_fixed_window


def test_markdown_headings_yield_section_chunks():
    text = "# Title\n\nintro\n\n## Methods\nWe did things.\n\n## Results\nIt worked."
    chunks = chunk_paper(text, paper_id="p1")
    sections = [c.section for c in chunks]
    assert "Methods" in sections and "Results" in sections


def test_numbered_headings_detected():
    text = (
        "1. Introduction\nPaper intro.\n\n"
        "2. Methods\nWe did things.\n\n"
        "3. Results\nIt worked."
    )
    chunks = chunk_paper(text)
    sections = [(c.section or "") for c in chunks]
    assert any("Introduction" in s for s in sections)
    assert any("Methods" in s for s in sections)
    assert any("Results" in s for s in sections)


def test_references_section_dropped_by_default():
    text = (
        "## Introduction\nIntro.\n\n"
        "## Methods\nMethods.\n\n"
        "## References\n[1] Smith et al."
    )
    chunks = chunk_paper(text)
    sections = [(c.section or "").lower() for c in chunks]
    assert "references" not in sections


def test_long_text_with_no_headings_falls_back_to_window():
    text = " ".join(["word"] * 2000)
    chunks = chunk_paper(text, paper_id="x", target_words=500, overlap_words=100)
    assert len(chunks) >= 3
    assert all(c.section is None for c in chunks)


def test_char_offsets_match_original_text():
    text = "## Introduction\nThe quick brown fox."
    chunks = chunk_paper(text)
    for c in chunks:
        assert text[c.char_start:c.char_end] == c.content


def test_paper_id_propagates_to_chunks():
    text = "## Methods\nWe did things."
    chunks = chunk_paper(text, paper_id="abc123")
    assert all(c.paper_id == "abc123" for c in chunks)


def test_chunk_by_section_returns_strings():
    text = "## A\nfirst body\n\n## B\nsecond body"
    out = chunk_by_section(text)
    assert isinstance(out, list) and all(isinstance(s, str) for s in out)


def test_chunk_fixed_window_overlaps():
    text = " ".join(str(i) for i in range(100))
    out = chunk_fixed_window(text, window_size=40, overlap=10)
    assert len(out) >= 3
