"""Shared pytest fixtures.

`fake_embedder` and `make_paper` let most tests avoid real API calls and
real PDFs. `isolated_paper_cache` redirects Paper's module-level cache
constants to tmp_path so tests never touch the project's `cache/` dir.
"""
from __future__ import annotations

import pathlib

import pytest

from research_rag import Paper
from research_rag.chunking import Chunk


@pytest.fixture
def fake_embedder():
    """Deterministic embedding callable; matches Embedder's call shape."""

    def fn(texts, input_type="document"):
        def vec(s: str) -> list[float]:
            v = [0.0] * 8
            for ch in s.lower():
                v[ord(ch) % 8] += 1.0
            return v

        if isinstance(texts, str):
            return vec(texts)
        return [vec(t) for t in texts]

    fn.embed = lambda t, input_type="document": fn(t, input_type=input_type)
    return fn


@pytest.fixture
def isolated_paper_cache(tmp_path, monkeypatch):
    """Redirect Paper's cache constants to tmp_path. Returns the paths dict."""
    parsed = tmp_path / "parsed"; parsed.mkdir()
    bm25 = tmp_path / "bm25"; bm25.mkdir()
    summaries = tmp_path / "summaries"; summaries.mkdir()
    monkeypatch.setattr("research_rag.paper.PARSED_CACHE", parsed)
    monkeypatch.setattr("research_rag.paper.BM25_CACHE", bm25)
    monkeypatch.setattr("research_rag.paper.SUMMARIES_CACHE", summaries)
    return {"root": tmp_path, "parsed": parsed, "bm25": bm25, "summaries": summaries}


@pytest.fixture
def make_paper(fake_embedder):
    """Factory: build a Paper without hitting hash_pdf or parse_pdf.

    chunks_data is a list of {"section": str|None, "content": str}; if
    omitted, _chunks stays None and is computed lazily.
    """

    def _make(
        pid,
        text,
        chunks_data=None,
        summary=None,
        title=None,
        embedder=None,
    ):
        p = Paper.__new__(Paper)
        p.source_path = pathlib.Path(f"/fake/{pid}.pdf")
        p.paper_id = pid
        p.title = title
        p.authors = None
        p.year = None
        p.summary_prompt_version = "v1"
        p._embedder = embedder if embedder is not None else fake_embedder
        p._ocr = False
        p._text = text
        if chunks_data is not None:
            p._chunks = [
                Chunk(
                    content=c["content"],
                    section=c.get("section"),
                    subsection=None,
                    chunk_index=i,
                    char_start=0,
                    char_end=len(c["content"]),
                    paper_id=pid,
                )
                for i, c in enumerate(chunks_data)
            ]
        else:
            p._chunks = None
        p._summary = summary
        p._chunk_index = None
        return p

    return _make
