"""Grade a single Q&A pair, or re-grade a saved transcript JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from anthropic import Anthropic

from src.research_rag import Grader
from src.research_rag.config import DEFAULT_GRADING_MODEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade a Q&A pair (or re-grade a saved transcript)."
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Path to a saved transcript JSON (from a pipeline run) to re-grade.",
    )
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--answer", type=str, default=None)
    parser.add_argument("--ground-truth", type=str, default=None)
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Retrieved context — inline string or path to a text file.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_GRADING_MODEL,
        help=f"Anthropic model for grading (default: {DEFAULT_GRADING_MODEL}).",
    )
    args = parser.parse_args(argv)

    if args.transcript:
        data = json.loads(args.transcript.read_text())
        question = data["question"]
        answer = data["answer"]
        ground_truth = data.get("ground_truth")
        context = data.get("retrieved_context")
    else:
        if not (args.question and args.answer):
            print(
                "Provide either --transcript or both --question and --answer",
                file=sys.stderr,
            )
            return 1
        question = args.question
        answer = args.answer
        ground_truth = args.ground_truth
        if args.context and Path(args.context).is_file():
            context = Path(args.context).read_text()
        else:
            context = args.context

    grader = Grader(client=Anthropic(), model=args.model)
    result = grader.grade(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        retrieved_context=context,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
