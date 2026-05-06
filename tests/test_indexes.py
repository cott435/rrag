"""Tests for VectorIndex, BM25Index, Retriever — core retrieval primitives."""
from __future__ import annotations

import pytest

from research_rag import BM25Index, Retriever, VectorIndex


def fake_embed_fn(texts):
    """Cheap deterministic vectors derived from char-mod frequencies."""

    def vec(s: str) -> list[float]:
        v = [0.0] * 4
        for ch in s.lower():
            v[ord(ch) % 4] += 1.0
        return v

    if isinstance(texts, str):
        return vec(texts)
    return [vec(t) for t in texts]


# ---- VectorIndex ----


def test_vector_index_search_returns_top_k():
    idx = VectorIndex(embedding_fn=fake_embed_fn)
    idx.add_documents([
        {"content": "alpha", "paper_id": "p1"},
        {"content": "beta", "paper_id": "p2"},
        {"content": "gamma alpha", "paper_id": "p3"},
    ])
    results = idx.search("alpha", k=2)
    assert len(results) == 2


def test_vector_index_corpus_filter_restricts_to_subset():
    idx = VectorIndex(embedding_fn=fake_embed_fn)
    idx.add_documents([
        {"content": "alpha", "paper_id": "p1"},
        {"content": "alpha beta", "paper_id": "p2"},
        {"content": "alpha gamma", "paper_id": "p3"},
    ])
    results = idx.search("alpha", k=10, corpus_filter={"p1", "p3"})
    assert {d["paper_id"] for d, _ in results} == {"p1", "p3"}


def test_vector_index_query_dim_mismatch_raises():
    idx = VectorIndex(embedding_fn=fake_embed_fn)
    idx.add_documents([{"content": "x", "paper_id": "p1"}])
    with pytest.raises(ValueError, match="dimension"):
        idx.search([1.0, 2.0], k=1)


def test_vector_index_save_load_round_trip(tmp_path):
    idx = VectorIndex(embedding_fn=fake_embed_fn)
    idx.add_documents([{"content": f"doc{i}", "paper_id": f"p{i}"} for i in range(3)])
    path = tmp_path / "vec.pkl"
    idx.save(path)
    loaded = VectorIndex.load(path, embedding_fn=fake_embed_fn)
    assert len(loaded) == 3
    assert {d["paper_id"] for d in loaded.documents} == {"p0", "p1", "p2"}


# ---- BM25Index ----


def test_bm25_finds_keyword_match():
    idx = BM25Index()
    idx.add_documents([
        {"content": "the protein binding site", "paper_id": "p1"},
        {"content": "robotic manipulation policy", "paper_id": "p2"},
    ])
    results = idx.search("protein binding", k=2)
    assert results
    # BM25 normalization sorts ascending, so the strongest match is last.
    assert results[-1][0]["paper_id"] == "p1"


def test_bm25_corpus_filter():
    idx = BM25Index()
    idx.add_documents([
        {"content": "alpha alpha alpha", "paper_id": "p1"},
        {"content": "alpha alpha", "paper_id": "p2"},
        {"content": "alpha", "paper_id": "p3"},
    ])
    results = idx.search("alpha", k=10, corpus_filter={"p2"})
    assert all(d["paper_id"] == "p2" for d, _ in results)


def test_bm25_save_load_round_trip(tmp_path):
    idx = BM25Index()
    idx.add_documents([
        {"content": "machine learning", "paper_id": "p1"},
        {"content": "deep learning", "paper_id": "p2"},
    ])
    _ = idx.search("learning", k=2)  # builds the index
    path = tmp_path / "bm.pkl"
    idx.save(path)
    loaded = BM25Index.load(path)
    assert len(loaded) == 2 and loaded._index_built
    assert len(loaded.search("learning", k=2)) == 2


# ---- Retriever ----


def test_retriever_fuses_dense_and_lexical():
    vec = VectorIndex(embedding_fn=fake_embed_fn)
    bm = BM25Index()
    docs = [
        {"content": "neural networks for vision", "paper_id": "p1"},
        {"content": "vision transformers", "paper_id": "p2"},
        {"content": "robotics control", "paper_id": "p3"},
    ]
    r = Retriever(bm, vec)
    r.add_documents(docs)
    paper_ids = [d["paper_id"] for d, _ in r.search("vision transformers", k=2)]
    assert "p3" not in paper_ids


def test_retriever_passes_corpus_filter_to_underlying_indexes():
    vec = VectorIndex(embedding_fn=fake_embed_fn)
    bm = BM25Index()
    r = Retriever(bm, vec)
    r.add_documents([
        {"content": "alpha", "paper_id": "p1"},
        {"content": "alpha", "paper_id": "p2"},
    ])
    results = r.search("alpha", k=2, corpus_filter={"p1"})
    assert all(d["paper_id"] == "p1" for d, _ in results)


def test_retriever_requires_at_least_one_index():
    with pytest.raises(ValueError):
        Retriever()
