"""Run the prompt optimization pipeline over a candidate set."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.research_rag import (
    Conversation,
    Embedder,
    Grader,
    LLMClient,
    PromptRegistry,
    make_corpus_tools,
    web_search_tool,
)
from src.research_rag.config import (
    DEFAULT_GRADING_MODEL,
    DEFAULT_INFERENCE_MODEL,
    PAPERS_DIR,
    ensure_dirs,
)
from src.research_rag.corpus import PaperCorpus
from src.research_rag.pipeline import EvalDataset, PromptOptimizationPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run prompt optimization: candidate prompts × eval items, graded."
    )
    parser.add_argument("--task", required=True, help="Prompt task name (e.g. qa).")
    parser.add_argument(
        "--candidates",
        required=True,
        help="Comma-separated versions: 'qa.v1,qa.v2,qa.v3' or 'v1,v2,v3'.",
    )
    parser.add_argument(
        "--eval-set", required=True, type=Path, help="Path to JSON eval dataset."
    )
    parser.add_argument("--papers-dir", type=Path, default=PAPERS_DIR)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_INFERENCE_MODEL,
        help=f"Anthropic model for the QA conversation (default: {DEFAULT_INFERENCE_MODEL}).",
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default=DEFAULT_GRADING_MODEL,
        help=f"Anthropic model for grading (default: {DEFAULT_GRADING_MODEL}).",
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable the hosted web_search tool for QA conversations.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ensure_dirs()

    if not args.eval_set.exists():
        print(f"Eval set not found: {args.eval_set}", file=sys.stderr)
        return 1

    dataset = EvalDataset.from_file(args.eval_set)

    registry = PromptRegistry()
    versions = [c.strip() for c in args.candidates.split(",") if c.strip()]
    candidates = []
    for v in versions:
        if "." in v:
            task, version = v.rsplit(".", 1)
        else:
            task, version = args.task, v
        candidates.append(registry.load(task, version))

    embedder = Embedder()
    corpus = PaperCorpus(embedder=embedder, papers_dir=args.papers_dir)
    corpus.discover()

    inference_llm = LLMClient(model=args.model)
    grader_llm = LLMClient(model=args.grader_model)
    base_tools = make_corpus_tools(corpus)
    if not args.no_web_search:
        base_tools.append(web_search_tool())

    def runner_factory(prompt):
        return Conversation(
            llm=inference_llm,
            corpus=corpus,
            system_prompt=prompt.system,
            tools=base_tools,
        )

    grader = Grader(llm=grader_llm)

    pipeline = PromptOptimizationPipeline(
        corpus=corpus,
        eval_dataset=dataset,
        grader=grader,
        candidate_prompts=candidates,
        runner_factory=runner_factory,
    )

    n_runs = len(candidates) * len(dataset.items)
    print(
        f"Running {len(candidates)} prompt(s) × {len(dataset.items)} item(s) "
        f"= {n_runs} QA call(s) + {n_runs} grading call(s)..."
    )
    result = pipeline.run()
    print(f"\nWrote run to: {result.run_dir}\n")
    print((result.run_dir / "leaderboard.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
