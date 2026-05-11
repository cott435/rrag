"""Tests for PaperCorpus: two-tier search and the no-summary fallback path."""
from __future__ import annotations

from src.research_rag import ChunkResult, PaperCorpus, StructuredSummary


def _summary(tldr, keywords):
    return StructuredSummary(
        tldr=tldr, problem="", method="", key_results="",
        limitations="", contributions=[], keywords=keywords,
    )


def test_search_chunks_two_tier_ranks_relevant_paper_first(
    fake_embedder, isolated_paper_cache, make_paper
):
    paper_a = make_paper(
        "a", text="...", title="Protein paper",
        chunks_data=[
            {"section": "Intro", "content": "Protein binding site prediction."},
            {"section": "Methods", "content": "ESM embeddings with CNN."},
        ],
        summary=_summary("Protein binding via ESM.", ["protein", "ESM", "binding"]),
    )
    paper_b = make_paper(
        "b", text="...", title="Robotics paper",
        chunks_data=[{"section": "Intro", "content": "Reinforcement learning for robots."}],
        summary=_summary("RL for manipulation.", ["robot", "RL"]),
    )
    corpus = PaperCorpus(embedder=fake_embedder)
    corpus.papers = {"a": paper_a, "b": paper_b}

    results = corpus.search_chunks("protein binding ESM", k_papers=2, k_chunks=3)
    assert results
    assert results[0].paper_id == "a"
    assert all(isinstance(r, ChunkResult) for r in results)


def test_search_chunks_falls_back_when_no_summaries(
    fake_embedder, isolated_paper_cache, make_paper
):
    paper_a = make_paper(
        "a", text="...",
        chunks_data=[{"section": "Intro", "content": "alpha alpha alpha"}],
    )
    paper_b = make_paper(
        "b", text="...",
        chunks_data=[{"section": "Intro", "content": "beta beta beta"}],
    )
    corpus = PaperCorpus(embedder=fake_embedder)
    corpus.papers = {"a": paper_a, "b": paper_b}

    assert corpus.summary_index is None  # no summaries → no summary index
    results = corpus.search_chunks("alpha", k_chunks=2)
    assert results
    assert all(isinstance(r, ChunkResult) for r in results)


def test_summary_index_returns_none_when_no_papers_summarized(fake_embedder, make_paper):
    paper = make_paper("a", text="...", chunks_data=[{"content": "x"}])
    corpus = PaperCorpus(embedder=fake_embedder)
    corpus.papers = {"a": paper}
    assert corpus.summary_index is None


def test_find_relevant_papers_uses_summary_index(
    fake_embedder, isolated_paper_cache, make_paper
):
    paper_a = make_paper(
        "a", text="...", title="A",
        chunks_data=[{"section": "I", "content": "alpha"}],
        summary=_summary("alpha summary text", ["alpha"]),
    )
    paper_b = make_paper(
        "b", text="...", title="B",
        chunks_data=[{"section": "I", "content": "beta"}],
        summary=_summary("beta summary text", ["beta"]),
    )
    corpus = PaperCorpus(embedder=fake_embedder)
    corpus.papers = {"a": paper_a, "b": paper_b}

    results = corpus.find_relevant_papers("alpha keyword query", k=1)
    assert len(results) == 1
    assert results[0].paper_id == "a"


def test_get_raises_for_unknown_paper(fake_embedder):
    import pytest
    corpus = PaperCorpus(embedder=fake_embedder)
    with pytest.raises(KeyError):
        corpus.get("nonexistent")
