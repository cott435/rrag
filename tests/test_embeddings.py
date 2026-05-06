"""Tests for Embedder caching and rate-limit retry."""
from __future__ import annotations

import pickle
from unittest.mock import MagicMock

from research_rag.embeddings import Embedder


def make_voyage_mock(vectors_per_call):
    """Mock voyage client; .embed(...) returns scripted lists of vectors."""
    it = iter(vectors_per_call)
    client = MagicMock()

    def embed(texts, model, input_type):
        result = MagicMock()
        result.embeddings = next(it)
        return result

    client.embed = embed
    return client


def test_embedder_caches_per_text(tmp_path):
    client = make_voyage_mock([
        [[1.0, 2.0]],   # first call: just "a"
        [[3.0, 4.0]],   # second call: just "b"
    ])
    emb = Embedder(client=client, model="voyage-test", cache_dir=tmp_path)

    assert emb.embed("a") == [1.0, 2.0]
    # Same text → cache hit → no new API call
    assert emb.embed("a") == [1.0, 2.0]
    # New text → 1 new API call
    assert emb.embed("b") == [3.0, 4.0]


def test_embedder_cache_key_includes_model_and_input_type(tmp_path):
    client = make_voyage_mock([
        [[1.0]],  # voyage-A document
        [[2.0]],  # voyage-A query
        [[3.0]],  # voyage-B document
    ])
    emb_a = Embedder(client=client, model="voyage-A", cache_dir=tmp_path)
    emb_b = Embedder(client=client, model="voyage-B", cache_dir=tmp_path)
    assert emb_a.embed("x", input_type="document") == [1.0]
    assert emb_a.embed("x", input_type="query") == [2.0]
    assert emb_b.embed("x", input_type="document") == [3.0]


def test_embedder_returns_list_for_list_input(tmp_path):
    client = make_voyage_mock([[[1.0], [2.0], [3.0]]])
    emb = Embedder(client=client, model="voyage-test", cache_dir=tmp_path)
    assert emb.embed(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]


def test_embedder_returns_scalar_for_scalar_input(tmp_path):
    client = make_voyage_mock([[[1.0, 2.0]]])
    emb = Embedder(client=client, model="voyage-test", cache_dir=tmp_path)
    assert emb.embed("a") == [1.0, 2.0]


def test_embedder_retries_on_rate_limit_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **kw: None)
    calls = {"n": 0}

    def embed(texts, model, input_type):
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("rate limit exceeded (429)")
        result = MagicMock(); result.embeddings = [[7.0]]
        return result

    client = MagicMock(); client.embed = embed
    emb = Embedder(client=client, model="voyage-test", cache_dir=tmp_path)
    assert emb.embed("x") == [7.0]
    assert calls["n"] == 3


def test_embedder_corrupt_cache_recovers(tmp_path):
    client = make_voyage_mock([[[7.0]]])
    emb = Embedder(client=client, model="voyage-test", cache_dir=tmp_path)
    key = emb._cache_key("x", "document")
    cache_path = emb._cache_path(key)
    cache_path.write_bytes(b"not a pickle")

    assert emb.embed("x") == [7.0]
    # And the corrupt file got replaced with a valid pickle.
    assert pickle.loads(cache_path.read_bytes()) == [7.0]
