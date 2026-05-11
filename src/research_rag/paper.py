"""Paper class — the central data object. Lazy throughout."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import Chunk, chunk_paper
from .config import (
    BM25_CACHE,
    PARSED_CACHE,
    SUMMARIES_CACHE,
)
from .embeddings import Embedder
from .indexes import BM25Index, Retriever, VectorIndex
from .ingest import hash_pdf, parse_pdf, tei_to_markdown

logger = logging.getLogger(__name__)


@dataclass
class StructuredSummary:
    tldr: str
    problem: str
    method: str
    key_results: str
    limitations: str
    contributions: list[str]
    keywords: list[str]

    def to_dict(self) -> dict:
        return {
            "tldr": self.tldr,
            "problem": self.problem,
            "method": self.method,
            "key_results": self.key_results,
            "limitations": self.limitations,
            "contributions": list(self.contributions),
            "keywords": list(self.keywords),
        }

    @classmethod
    def from_dict(cls, data: dict) -> StructuredSummary:
        fields = (
            "tldr",
            "problem",
            "method",
            "key_results",
            "limitations",
            "contributions",
            "keywords",
        )
        return cls(**{f: data[f] for f in fields})


class Paper:
    """Lazy paper object. Each property checks cache before computing.

    Cache filenames are keyed on paper_id (a 16-char prefix of the source
    PDF's sha256). Summary cache files additionally include the
    summarization prompt version, so different prompts don't overwrite
    each other's outputs (important for the prompt-optimization pipeline).
    """

    def __init__(
        self,
        source_path: Path,
        embedder: Embedder | None = None,
        summary_prompt_version: str = "v1",
        title: str | None = None,
        authors: list[str] | None = None,
        year: int | None = None,
    ):
        self.source_path = Path(source_path)
        self.paper_id = hash_pdf(self.source_path)
        self.title = title
        self.authors = authors
        self.year = year
        self.summary_prompt_version = summary_prompt_version

        self._embedder = embedder

        self._tei_xml: str | None = None
        self._text: str | None = None
        self._chunks: list[Chunk] | None = None
        self._summary: StructuredSummary | None = None
        self._chunk_index: Retriever | None = None

    @property
    def tei_xml(self) -> str:
        if self._tei_xml is not None:
            return self._tei_xml
        cache_path = PARSED_CACHE / f"{self.paper_id}.tei.xml"
        if cache_path.exists():
            self._tei_xml = cache_path.read_text(encoding="utf-8")
            return self._tei_xml
        xml = parse_pdf(self.source_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(xml, encoding="utf-8")
        self._tei_xml = xml
        return self._tei_xml

    @property
    def text(self) -> str:
        if self._text is not None:
            return self._text
        cache_path = PARSED_CACHE / f"{self.paper_id}.md"
        if cache_path.exists():
            self._text = cache_path.read_text(encoding="utf-8")
            return self._text
        text = tei_to_markdown(self.tei_xml)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        self._text = text
        return self._text

    @property
    def chunks(self) -> list[Chunk]:
        if self._chunks is not None:
            return self._chunks
        self._chunks = chunk_paper(self.tei_xml, paper_id=self.paper_id)
        return self._chunks

    def _summary_path(self, prompt_version: str | None = None) -> Path:
        v = prompt_version or self.summary_prompt_version
        return SUMMARIES_CACHE / f"{self.paper_id}.{v}.json"

    @property
    def summary(self) -> StructuredSummary:
        return self.get_summary()

    def get_summary(self, prompt_version: str | None = None) -> StructuredSummary:
        """Load a summary from cache, optionally for a specific prompt version."""
        v = prompt_version or self.summary_prompt_version
        if self._summary is not None and v == self.summary_prompt_version:
            return self._summary
        cache_path = self._summary_path(v)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                summary = StructuredSummary.from_dict(data)
                if v == self.summary_prompt_version:
                    self._summary = summary
                return summary
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Corrupt summary cache %s; removing. %s", cache_path, e)
                cache_path.unlink(missing_ok=True)
        raise ValueError(
            f"No summary cached for paper {self.paper_id} at version {v}. "
            f"Run Summarizer.summarize() first."
        )

    def has_summary(self, prompt_version: str | None = None) -> bool:
        if self._summary is not None and prompt_version in (None, self.summary_prompt_version):
            return True
        return self._summary_path(prompt_version).exists()

    def set_summary(
        self,
        summary: StructuredSummary,
        prompt_version: str | None = None,
    ) -> None:
        v = prompt_version or self.summary_prompt_version
        cache_path = self._summary_path(v)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**summary.to_dict(), "_prompt_version": v}
        cache_path.write_text(json.dumps(payload, indent=2))
        if v == self.summary_prompt_version:
            self._summary = summary

    # ----- chunk index -----

    def _build_documents(self) -> list[dict[str, Any]]:
        return [
            {
                "content": c.content,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "paper_id": self.paper_id,
            }
            for c in self.chunks
        ]

    @property
    def chunk_index(self) -> Retriever:
        if self._chunk_index is not None:
            return self._chunk_index
        if self._embedder is None:
            raise ValueError(
                f"Paper {self.paper_id} cannot build chunk_index without an Embedder. "
                f"Pass embedder= when constructing the Paper."
            )

        bm25_cache = BM25_CACHE / f"{self.paper_id}.pkl"

        bm: BM25Index | None = None
        if bm25_cache.exists():
            try:
                bm = BM25Index.load(bm25_cache)
            except Exception as e:
                logger.warning(
                    "Corrupt BM25 cache for %s; rebuilding. %s", self.paper_id, e
                )
                bm25_cache.unlink(missing_ok=True)
                bm = None

        if bm is None:
            documents = self._build_documents()
            bm = BM25Index()
            bm.add_documents(documents)
            bm.save(bm25_cache)
        else:
            # Reuse BM25's stored docs so RRF id() matching works across indexes
            documents = bm.documents

        # VectorIndex isn't pickled per-paper; per-chunk vectors are cached
        # by the Embedder, so rebuilding from documents is cheap.
        vec = VectorIndex(embedding_fn=self._embedder)
        vec.add_documents(documents)

        self._chunk_index = Retriever(bm, vec)
        return self._chunk_index

    def clear_cache(self) -> None:
        """Remove cached parsed text, BM25 index, and summaries for this paper.

        Per-chunk embeddings under EMBEDDINGS_CACHE are keyed by chunk
        content hash (not paper_id), so they're shared across re-runs and
        not cleared here. To force re-embedding too, manually clear that dir.
        """
        (PARSED_CACHE / f"{self.paper_id}.md").unlink(missing_ok=True)
        (PARSED_CACHE / f"{self.paper_id}.tei.xml").unlink(missing_ok=True)
        (BM25_CACHE / f"{self.paper_id}.pkl").unlink(missing_ok=True)
        for summary_file in SUMMARIES_CACHE.glob(f"{self.paper_id}.*.json"):
            summary_file.unlink(missing_ok=True)
        self._tei_xml = None
        self._text = None
        self._chunks = None
        self._summary = None
        self._chunk_index = None

    def __repr__(self) -> str:
        return f"Paper(id={self.paper_id}, source={self.source_path.name})"
