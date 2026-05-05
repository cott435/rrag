"""Hand-rolled hybrid retrieval: dense (Voyage) + BM25 fused with RRF.

Ports VectorIndex / BM25Index / Retriever from chunking_embed.ipynb and
adds: pickle save/load on each index type, and a corpus_filter parameter
on search() that restricts results to a subset of paper_ids.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Protocol


class SearchIndex(Protocol):
    def add_document(self, document: dict[str, Any]) -> None: ...
    def add_documents(self, documents: list[dict[str, Any]]) -> None: ...
    def search(
        self,
        query: Any,
        k: int = 1,
        corpus_filter: set[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]: ...


class VectorIndex:
    """Dense vector index. Cosine or euclidean over Voyage embeddings."""

    def __init__(
        self,
        distance_metric: str = "cosine",
        embedding_fn: Callable[..., Any] | None = None,
    ):
        if distance_metric not in ("cosine", "euclidean"):
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self.vectors: list[list[float]] = []
        self.documents: list[dict[str, Any]] = []
        self._vector_dim: int | None = None
        self._distance_metric = distance_metric
        self._embedding_fn = embedding_fn

    def add_document(self, document: dict[str, Any]) -> None:
        if self._embedding_fn is None:
            raise ValueError("Embedding function not provided during initialization.")
        self._validate_doc(document)
        vector = self._embedding_fn(document["content"])
        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        if self._embedding_fn is None:
            raise ValueError("Embedding function not provided during initialization.")
        if not isinstance(documents, list):
            raise TypeError("Documents must be a list of dictionaries.")
        if not documents:
            return
        for i, doc in enumerate(documents):
            self._validate_doc(doc, idx=i)
        contents = [doc["content"] for doc in documents]
        vectors = self._embedding_fn(contents)
        for vector, document in zip(vectors, documents):
            self.add_vector(vector=vector, document=document)

    def add_vector(self, vector: list[float], document: dict[str, Any]) -> None:
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) for x in vector
        ):
            raise TypeError("Vector must be a list of numbers.")
        self._validate_doc(document)
        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(
                f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}"
            )
        self.vectors.append(list(vector))
        self.documents.append(document)

    def search(
        self,
        query: Any,
        k: int = 1,
        corpus_filter: set[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        if not self.vectors:
            return []
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if isinstance(query, str):
            if self._embedding_fn is None:
                raise ValueError("Embedding function not provided for string query.")
            query_vector = self._embedding_fn(query)
        elif isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            query_vector = query
        else:
            raise TypeError("Query must be a string or a list of numbers.")

        if self._vector_dim is None or len(query_vector) != self._vector_dim:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self._vector_dim}, "
                f"got {len(query_vector)}"
            )

        dist_func = (
            self._cosine_distance
            if self._distance_metric == "cosine"
            else self._euclidean_distance
        )

        scored: list[tuple[float, dict[str, Any]]] = []
        for i, stored in enumerate(self.vectors):
            doc = self.documents[i]
            if corpus_filter is not None and doc.get("paper_id") not in corpus_filter:
                continue
            scored.append((dist_func(query_vector, stored), doc))

        scored.sort(key=lambda item: item[0])
        return [(doc, dist) for dist, doc in scored[:k]]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "vectors": self.vectors,
            "documents": self.documents,
            "_vector_dim": self._vector_dim,
            "_distance_metric": self._distance_metric,
        }
        with path.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(
        cls,
        path: Path,
        embedding_fn: Callable[..., Any] | None = None,
    ) -> VectorIndex:
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        idx = cls(distance_metric=state["_distance_metric"], embedding_fn=embedding_fn)
        idx.vectors = state["vectors"]
        idx.documents = state["documents"]
        idx._vector_dim = state["_vector_dim"]
        return idx

    @staticmethod
    def _validate_doc(document: dict[str, Any], idx: int | None = None) -> None:
        prefix = f"Document at index {idx} " if idx is not None else "Document "
        if not isinstance(document, dict):
            raise TypeError(f"{prefix}must be a dictionary.")
        if "content" not in document:
            raise ValueError(f"{prefix}must contain a 'content' key.")
        if not isinstance(document["content"], str):
            raise TypeError(f"{prefix}'content' must be a string.")

    @staticmethod
    def _euclidean_distance(v1: list[float], v2: list[float]) -> float:
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(v1, v2)))

    @staticmethod
    def _dot_product(v1: list[float], v2: list[float]) -> float:
        return sum(p * q for p, q in zip(v1, v2))

    @staticmethod
    def _magnitude(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    def _cosine_distance(self, v1: list[float], v2: list[float]) -> float:
        m1, m2 = self._magnitude(v1), self._magnitude(v2)
        if m1 == 0 and m2 == 0:
            return 0.0
        if m1 == 0 or m2 == 0:
            return 1.0
        cos_sim = self._dot_product(v1, v2) / (m1 * m2)
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return 1.0 - cos_sim

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        has_fn = "yes" if self._embedding_fn else "no"
        return (
            f"VectorIndex(count={len(self)}, dim={self._vector_dim}, "
            f"metric='{self._distance_metric}', has_embedding_fn={has_fn})"
        )


class BM25Index:
    """Lexical BM25 index. Okapi formulation with RRF-friendly normalization."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, Any]] = []
        self._corpus_tokens: list[list[str]] = []
        self._doc_len: list[int] = []
        self._doc_freqs: dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: dict[str, float] = {}
        self._index_built: bool = False
        self._tokenizer = tokenizer if tokenizer else self._default_tokenizer

    @staticmethod
    def _default_tokenizer(text: str) -> list[str]:
        return [t for t in re.split(r"\W+", text.lower()) if t]

    def add_document(self, document: dict[str, Any]) -> None:
        self._validate_doc(document)
        tokens = self._tokenizer(document["content"])
        self.documents.append(document)
        self._corpus_tokens.append(tokens)
        self._update_stats_add(tokens)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        if not isinstance(documents, list):
            raise TypeError("Documents must be a list of dictionaries.")
        if not documents:
            return
        for i, doc in enumerate(documents):
            self._validate_doc(doc, idx=i)
            tokens = self._tokenizer(doc["content"])
            self.documents.append(doc)
            self._corpus_tokens.append(tokens)
            self._update_stats_add(tokens)
        self._index_built = False

    def _update_stats_add(self, doc_tokens: list[str]) -> None:
        self._doc_len.append(len(doc_tokens))
        seen: set[str] = set()
        for token in doc_tokens:
            if token not in seen:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen.add(token)
        self._index_built = False

    def _calculate_idf(self) -> None:
        N = len(self.documents)
        self._idf = {
            term: math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            for term, freq in self._doc_freqs.items()
        }

    def _build_index(self) -> None:
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return
        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        self._calculate_idf()
        self._index_built = True

    def _compute_bm25_score(self, query_tokens: list[str], doc_index: int) -> float:
        score = 0.0
        doc_term_counts = Counter(self._corpus_tokens[doc_index])
        doc_length = self._doc_len[doc_index]
        for token in query_tokens:
            if token not in self._idf:
                continue
            idf = self._idf[token]
            tf = doc_term_counts.get(token, 0)
            num = idf * tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * (doc_length / self._avg_doc_len))
            score += num / (den + 1e-9)
        return score

    def search(
        self,
        query: Any,
        k: int = 1,
        corpus_filter: set[str] | None = None,
        score_normalization_factor: float = 0.1,
    ) -> list[tuple[dict[str, Any], float]]:
        if not self.documents:
            return []
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if not isinstance(query, str):
            raise TypeError("Query must be a string for BM25Index.")

        if not self._index_built:
            self._build_index()
        if self._avg_doc_len == 0:
            return []

        tokens = self._tokenizer(query)
        if not tokens:
            return []

        raw: list[tuple[float, dict[str, Any]]] = []
        for i in range(len(self.documents)):
            doc = self.documents[i]
            if corpus_filter is not None and doc.get("paper_id") not in corpus_filter:
                continue
            score = self._compute_bm25_score(tokens, i)
            if score > 1e-9:
                raw.append((score, doc))

        raw.sort(key=lambda item: item[0], reverse=True)
        normalized = [
            (doc, math.exp(-score_normalization_factor * raw_score))
            for raw_score, doc in raw[:k]
        ]
        normalized.sort(key=lambda item: item[1])
        return normalized

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
            "_corpus_tokens": self._corpus_tokens,
            "_doc_len": self._doc_len,
            "_doc_freqs": self._doc_freqs,
            "_avg_doc_len": self._avg_doc_len,
            "_idf": self._idf,
            "_index_built": self._index_built,
        }
        with path.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(
        cls,
        path: Path,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> BM25Index:
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        idx = cls(k1=state["k1"], b=state["b"], tokenizer=tokenizer)
        idx.documents = state["documents"]
        idx._corpus_tokens = state["_corpus_tokens"]
        idx._doc_len = state["_doc_len"]
        idx._doc_freqs = state["_doc_freqs"]
        idx._avg_doc_len = state["_avg_doc_len"]
        idx._idf = state["_idf"]
        idx._index_built = state["_index_built"]
        return idx

    @staticmethod
    def _validate_doc(document: dict[str, Any], idx: int | None = None) -> None:
        prefix = f"Document at index {idx} " if idx is not None else "Document "
        if not isinstance(document, dict):
            raise TypeError(f"{prefix}must be a dictionary.")
        if "content" not in document:
            raise ValueError(f"{prefix}must contain a 'content' key.")
        if not isinstance(document["content"], str):
            raise TypeError(f"{prefix}'content' must be a string.")

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return (
            f"BM25Index(count={len(self)}, k1={self.k1}, b={self.b}, "
            f"index_built={self._index_built})"
        )


class Retriever:
    """Hybrid retriever fusing one or more SearchIndex instances via RRF.

    RRF matches documents across indexes via id() — this works as long as
    all indexes were populated with the same dict objects (use the same
    list when calling add_documents on multiple indexes, or re-use the
    docs from a loaded index).
    """

    def __init__(self, *indexes: SearchIndex):
        if not indexes:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)

    def add_document(self, document: dict[str, Any]) -> None:
        for idx in self._indexes:
            idx.add_document(document)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        for idx in self._indexes:
            idx.add_documents(documents)

    def search(
        self,
        query_text: str,
        k: int = 1,
        k_rrf: int = 60,
        corpus_filter: set[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k_rrf < 0:
            raise ValueError("k_rrf must be non-negative.")

        per_index_results = [
            idx.search(query_text, k=k * 5, corpus_filter=corpus_filter)
            for idx in self._indexes
        ]

        ranks: dict[int, dict[str, Any]] = {}
        for i, results in enumerate(per_index_results):
            for rank, (doc, _) in enumerate(results):
                key = id(doc)
                if key not in ranks:
                    ranks[key] = {
                        "doc": doc,
                        "ranks": [float("inf")] * len(self._indexes),
                    }
                ranks[key]["ranks"][i] = rank + 1

        def rrf(rs: list[float]) -> float:
            return sum(1.0 / (k_rrf + r) for r in rs if r != float("inf"))

        scored = [(entry["doc"], rrf(entry["ranks"])) for entry in ranks.values()]
        scored = [(d, s) for d, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
