"""Batch-ingest all PDFs in papers/. Parses (via GROBID), chunks, embeds, and caches per-paper indexes."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.research_rag import Embedder
from src.research_rag.config import PAPERS_DIR, ensure_dirs
from src.research_rag.corpus import PaperCorpus
from src.research_rag.grobid_docker import GrobidDocker, GrobidVariant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest all PDFs in the papers directory: parse via GROBID, chunk, embed, cache."
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=PAPERS_DIR,
        help=f"Directory of PDFs to ingest (default: {PAPERS_DIR}).",
    )
    parser.add_argument(
        "--grobid-variant",
        choices=[v.value for v in GrobidVariant],
        default=GrobidVariant.CRF.value,
        help="GROBID image variant (default: crf).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Pass --gpus all to docker run (only valid with --grobid-variant full).",
    )
    parser.add_argument(
        "--grobid-port",
        type=int,
        default=8070,
        help="Host port to expose GROBID on (default: 8070).",
    )
    parser.add_argument(
        "--no-start-grobid",
        action="store_true",
        help="Skip auto-starting the GROBID container; assume one is already running at RAG_GROBID_URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if cached. Clears parsed/BM25/summary caches per paper.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ensure_dirs()

    pdfs = sorted(args.papers_dir.glob("*.pdf")) + sorted(args.papers_dir.glob("*.PDF"))
    if not pdfs:
        print(f"No PDFs found in {args.papers_dir}")
        return 0

    os.environ["RAG_GROBID_URL"] = f"http://localhost:{args.grobid_port}"

    embedder = Embedder()
    corpus = PaperCorpus(embedder=embedder, papers_dir=args.papers_dir)

    print(f"Ingesting {len(pdfs)} PDF(s) from {args.papers_dir}")

    if args.no_start_grobid:
        return _run_ingest(corpus, pdfs, force=args.force)

    grobid = GrobidDocker(
        variant=GrobidVariant(args.grobid_variant),
        gpu=args.gpu,
        port=args.grobid_port,
    )
    with grobid:
        return _run_ingest(corpus, pdfs, force=args.force)


def _run_ingest(corpus: PaperCorpus, pdfs: list[Path], *, force: bool) -> int:
    failures: list[tuple[Path, Exception]] = []
    for pdf_path in pdfs:
        paper = corpus.add_paper(pdf_path)
        prefix = f"  {pdf_path.name} -> {paper.paper_id}"
        if force:
            paper.clear_cache()
        try:
            _ = paper.text
            _ = paper.chunks
            _ = paper.chunk_index
            sections = sum(1 for c in paper.chunks if c.section)
            print(f"{prefix}  chunks={len(paper.chunks)} sections={sections}")
        except Exception as e:
            failures.append((pdf_path, e))
            print(f"{prefix}  FAILED: {e}", file=sys.stderr)

    print(f"\nDone: {len(pdfs) - len(failures)} ingested, {len(failures)} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
