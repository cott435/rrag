"""PaperCorpus — manages all papers and provides corpus-wide retrieval."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import PAPERS_DIR
from .embeddings import Embedder
from .indexes import BM25Index, Retriever, VectorIndex
from .paper import Paper, StructuredSummary

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    paper_id: str
    paper_title: str | None
    section: str | None
    chunk_index: int
    content: str
    score: float


class PaperCorpus:
    """Collection of Papers with corpus-level summary index + two-tier search.

    The summary index is built lazily from per-paper cached summaries.
    Papers without cached summaries are excluded from the summary index
    but still searchable via the chunk-level fallback in search_chunks.
    """

    def __init__(
        self,
        embedder: Embedder,
        papers_dir: Path = PAPERS_DIR,
    ):
        self.embedder = embedder
        self.papers_dir = Path(papers_dir)
        self.papers: dict[str, Paper] = {}
        self._summary_index: Retriever | None = None
        self._summary_index_paper_ids: frozenset[str] | None = None

    # ----- paper management -----

    def discover(self, ocr: bool = False) -> list[Paper]:
        """Find all PDFs in papers_dir and register them as Paper objects."""
        found: list[Paper] = []
        pdfs = sorted(self.papers_dir.glob("*.pdf")) + sorted(
            self.papers_dir.glob("*.PDF")
        )
        for pdf in pdfs:
            found.append(self.add_paper(pdf, ocr=ocr))
        return found

    def add_paper(self, pdf_path: Path, ocr: bool = False) -> Paper:
        paper = Paper(pdf_path, embedder=self.embedder, ocr=ocr)
        if paper.paper_id in self.papers:
            return self.papers[paper.paper_id]
        self.papers[paper.paper_id] = paper
        # Invalidate summary index since the paper set changed.
        self._summary_index = None
        self._summary_index_paper_ids = None
        return paper

    def get(self, paper_id: str) -> Paper:
        if paper_id not in self.papers:
            raise KeyError(f"Paper not found in corpus: {paper_id}")
        return self.papers[paper_id]

    # ----- summary index -----

    @staticmethod
    def _summary_text(s: StructuredSummary) -> str:
        """Render a StructuredSummary into text for embedding/BM25."""
        keywords = ", ".join(s.keywords)
        return (
            f"{s.tldr}\n\n"
            f"Problem: {s.problem}\n\n"
            f"Method: {s.method}\n\n"
            f"Key results: {s.key_results}\n\n"
            f"Keywords: {keywords}"
        )

    def _papers_with_summaries(self) -> list[Paper]:
        return [p for p in self.papers.values() if p.has_summary()]

    @property
    def summary_index(self) -> Retriever | None:
        """Retriever over papers' summaries, or None if none are cached.

        Rebuilt lazily when the set of summarized papers changes.
        """
        eligible = self._papers_with_summaries()
        if not eligible:
            return None

        eligible_ids = frozenset(p.paper_id for p in eligible)
        if (
            self._summary_index is not None
            and self._summary_index_paper_ids == eligible_ids
        ):
            return self._summary_index

        documents = []
        for paper in eligible:
            try:
                summary = paper.summary
            except ValueError:
                continue
            documents.append(
                {
                    "content": self._summary_text(summary),
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                }
            )

        if not documents:
            return None

        vec = VectorIndex(embedding_fn=self.embedder)
        bm = BM25Index()
        retriever = Retriever(bm, vec)
        retriever.add_documents(documents)

        self._summary_index = retriever
        self._summary_index_paper_ids = eligible_ids
        return retriever

    # ----- search -----

    def find_relevant_papers(self, query: str, k: int = 5) -> list[Paper]:
        """Search the summary index for papers relevant to query.

        If no summary index exists, returns all papers in the corpus
        (caller can then search chunks across them).
        """
        if self.summary_index is None:
            return list(self.papers.values())
        results = self.summary_index.search(query, k=k)
        papers: list[Paper] = []
        for doc, _ in results:
            pid = doc["paper_id"]
            if pid in self.papers:
                papers.append(self.papers[pid])
        return papers

    def search_chunks(
        self,
        query: str,
        k_papers: int = 5,
        k_chunks: int = 10,
        k_rrf: int = 60,
    ) -> list[ChunkResult]:
        """Two-tier retrieval: filter by summary, then per-paper chunk search.

        When the summary index exists, candidates are top k_papers by
        summary relevance, and each chunk's final rank fuses its
        per-paper chunk rank with its paper's summary rank via RRF.

        Without summaries, fans out to every paper's chunk index and
        ranks by chunk score alone.
        """
        if not self.papers:
            return []

        if self.summary_index is None:
            return self._search_chunks_fallback(query, k_chunks=k_chunks)

        summary_results = self.summary_index.search(query, k=k_papers)
        if not summary_results:
            return self._search_chunks_fallback(query, k_chunks=k_chunks)

        per_paper_k = max(1, k_chunks)
        scored: list[tuple[dict, float]] = []
        for sum_rank_idx, (sum_doc, _) in enumerate(summary_results):
            paper_id = sum_doc["paper_id"]
            if paper_id not in self.papers:
                continue
            paper = self.papers[paper_id]
            try:
                chunk_results = paper.chunk_index.search(query, k=per_paper_k)
            except Exception as e:
                logger.warning("Chunk search failed for paper %s: %s", paper_id, e)
                continue
            summary_rank = sum_rank_idx + 1
            for chunk_rank_idx, (chunk_doc, _) in enumerate(chunk_results):
                chunk_rank = chunk_rank_idx + 1
                rrf = 1.0 / (k_rrf + summary_rank) + 1.0 / (k_rrf + chunk_rank)
                scored.append((chunk_doc, rrf))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._to_chunk_result(d, s) for d, s in scored[:k_chunks]]

    def _search_chunks_fallback(self, query: str, k_chunks: int) -> list[ChunkResult]:
        """Fan out across all paper chunk indexes when no summary tier exists."""
        scored: list[tuple[dict, float]] = []
        for paper in self.papers.values():
            try:
                results = paper.chunk_index.search(query, k=k_chunks)
            except Exception as e:
                logger.warning(
                    "Chunk search failed for paper %s: %s", paper.paper_id, e
                )
                continue
            scored.extend(results)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._to_chunk_result(d, s) for d, s in scored[:k_chunks]]

    def _to_chunk_result(self, doc: dict, score: float) -> ChunkResult:
        paper = self.papers.get(doc["paper_id"])
        return ChunkResult(
            paper_id=doc["paper_id"],
            paper_title=paper.title if paper else None,
            section=doc.get("section"),
            chunk_index=doc["chunk_index"],
            content=doc["content"],
            score=score,
        )

    def __len__(self) -> int:
        return len(self.papers)

    def __repr__(self) -> str:
        with_summaries = sum(1 for p in self.papers.values() if p.has_summary())
        return (
            f"PaperCorpus(papers={len(self.papers)}, "
            f"with_summaries={with_summaries})"
        )
