"""Summarize one or all papers in the corpus."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.research_rag import Embedder, LLMClient, Paper
from src.research_rag.config import (
    DEFAULT_BULK_MODEL,
    DEFAULT_INFERENCE_MODEL,
    PAPERS_DIR,
    ensure_dirs,
    DEFAULT_LOCAL_MODEL
)
from src.research_rag.corpus import PaperCorpus
from src.research_rag.summarizer import Summarizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize papers in the corpus. Lazy by default; --force bypasses cache."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--paper",
        type=str,
        help="paper_id (exact) or filename substring (case-insensitive).",
    )
    target.add_argument("--all", action="store_true", help="Summarize all papers in papers-dir.")

    parser.add_argument("--papers-dir", type=Path, default=PAPERS_DIR)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LOCAL_MODEL,
        help=(
            f"Model id (default: {DEFAULT_INFERENCE_MODEL}). "
            f"Use {DEFAULT_BULK_MODEL} for cheap bulk runs, or a local id "
            "(e.g. 'qwen3:1.7b') to route through Ollama."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Bypass summary cache.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(["--all"])
    #args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ensure_dirs()

    embedder = Embedder()
    corpus = PaperCorpus(embedder=embedder, papers_dir=args.papers_dir)
    corpus.discover()

    if not corpus.papers:
        print(f"No papers found in {args.papers_dir}")
        return 0

    if args.all:
        targets: list[Paper] = list(corpus.papers.values())
    else:
        targets = _resolve_target(corpus, args.paper)
        if not targets:
            print(f"No paper matching '{args.paper}'", file=sys.stderr)
            return 1
        if len(targets) > 1:
            print(f"Ambiguous match — '{args.paper}' matches {len(targets)} papers:", file=sys.stderr)
            for p in targets:
                print(f"  {p.paper_id}  {p.source_path.name}", file=sys.stderr)
            return 1

    summarizer = Summarizer(llm=LLMClient(model=args.model))
    print(f"Using prompt {summarizer.prompt.name}.{summarizer.prompt.version} | model={args.model}")

    failures = 0
    for paper in targets:
        print(f"\n=== {paper.paper_id}  {paper.source_path.name} ===")
        try:
            summary = summarizer.summarize(paper, force=args.force)
            print(json.dumps(summary.to_dict(), indent=2))
        except Exception as e:
            failures += 1
            print(f"FAILED: {e}", file=sys.stderr)

    return 1 if failures else 0


def _resolve_target(corpus: PaperCorpus, target: str) -> list[Paper]:
    """Match by paper_id (exact) or filename (case-insensitive substring)."""
    if target in corpus.papers:
        return [corpus.papers[target]]
    return [
        p for p in corpus.papers.values()
        if target.lower() in p.source_path.name.lower()
    ]


if __name__ == "__main__":
    sys.exit(main())
