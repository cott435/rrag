"""PromptOptimizationPipeline — runs candidate prompts × eval questions, grades, ranks."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import EVAL_RUNS
from .conversation import Conversation
from .corpus import PaperCorpus
from .grader import GradeResult, Grader
from .prompts import PromptTemplate

logger = logging.getLogger(__name__)


@dataclass
class EvalItem:
    id: str
    question: str
    ground_truth: str | None = None
    relevant_paper_ids: list[str] = field(default_factory=list)
    category: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> EvalItem:
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            ground_truth=data.get("ground_truth"),
            relevant_paper_ids=list(data.get("relevant_paper_ids", [])),
            category=data.get("category"),
        )


@dataclass
class EvalDataset:
    name: str
    items: list[EvalItem]

    @classmethod
    def from_file(cls, path: Path) -> EvalDataset:
        data = json.loads(Path(path).read_text())
        return cls(
            name=str(data.get("name", Path(path).stem)),
            items=[EvalItem.from_dict(d) for d in data["items"]],
        )


@dataclass
class RunItemResult:
    item_id: str
    prompt_version: str
    question: str
    ground_truth: str | None
    answer: str
    grade: GradeResult | None
    error: str | None = None
    category: str | None = None


@dataclass
class PromptStats:
    prompt_version: str
    n_items: int
    n_failures: int
    mean_overall: float | None
    mean_faithfulness: float | None
    mean_relevance: float | None
    mean_completeness: float | None
    mean_citation_quality: float | None
    mean_calibration: float | None
    failure_modes_counts: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "prompt_version": self.prompt_version,
            "n_items": self.n_items,
            "n_failures": self.n_failures,
            "mean_overall": self.mean_overall,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_relevance": self.mean_relevance,
            "mean_completeness": self.mean_completeness,
            "mean_citation_quality": self.mean_citation_quality,
            "mean_calibration": self.mean_calibration,
            "failure_modes_counts": dict(self.failure_modes_counts),
        }


@dataclass
class RunResult:
    run_dir: Path
    stats: list[PromptStats]
    items: list[RunItemResult]


class PromptOptimizationPipeline:
    """Runs candidate prompts × eval items, grades each answer, writes a run dir.

    Each (prompt, item) generates a fresh Conversation via runner_factory.
    The conversation's tool_result blocks for retrieve_from_papers are
    extracted and passed to the grader as retrieved_context, so the
    grader can flag ignored_context / hallucination against what the
    model actually saw.

    Run output (immutable, written under runs_dir/{ISO_TIMESTAMP}/):
      - {prompt}/{item_id}.json — per-(prompt,item) transcript
      - {prompt}/prompt.md — archived copy of the prompt template
      - leaderboard.md — markdown leaderboard
      - summary.json — machine-readable aggregate
    """

    def __init__(
        self,
        corpus: PaperCorpus,
        eval_dataset: EvalDataset,
        grader: Grader,
        candidate_prompts: list[PromptTemplate],
        runner_factory: Callable[[PromptTemplate], Conversation],
        runs_dir: Path = EVAL_RUNS,
    ):
        if not candidate_prompts:
            raise ValueError("Need at least one candidate prompt")
        if not eval_dataset.items:
            raise ValueError("Eval dataset is empty")
        self.corpus = corpus
        self.eval_dataset = eval_dataset
        self.grader = grader
        self.candidate_prompts = list(candidate_prompts)
        self.runner_factory = runner_factory
        self.runs_dir = Path(runs_dir)

    def run(self) -> RunResult:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run %s started in %s", timestamp, run_dir)

        all_results: list[RunItemResult] = []
        for prompt in self.candidate_prompts:
            prompt_dir = run_dir / f"{prompt.name}.{prompt.version}"
            prompt_dir.mkdir(exist_ok=True)
            (prompt_dir / "prompt.md").write_text(prompt.to_string(), encoding="utf-8")
            for item in self.eval_dataset.items:
                logger.info("Prompt %s | item %s", prompt.version, item.id)
                result = self._run_one(prompt, item, prompt_dir)
                all_results.append(result)

        stats = self._aggregate(all_results)
        leaderboard = self._format_leaderboard(stats, all_results)
        (run_dir / "leaderboard.md").write_text(leaderboard, encoding="utf-8")
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "dataset": self.eval_dataset.name,
                    "timestamp": timestamp,
                    "n_items": len(self.eval_dataset.items),
                    "candidates": [
                        f"{p.name}.{p.version}" for p in self.candidate_prompts
                    ],
                    "stats": [s.to_dict() for s in stats],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Run %s complete: %s", timestamp, run_dir / "leaderboard.md")
        return RunResult(run_dir=run_dir, stats=stats, items=all_results)

    def _run_one(
        self,
        prompt: PromptTemplate,
        item: EvalItem,
        prompt_dir: Path,
    ) -> RunItemResult:
        answer = ""
        retrieved_context: list[str] = []
        conv: Conversation | None = None
        run_error: str | None = None
        grade: GradeResult | None = None
        grade_error: str | None = None

        try:
            conv = self.runner_factory(prompt)
            answer = conv.ask(item.question)
            retrieved_context = _extract_retrieved_context(conv.messages)
        except Exception as e:
            logger.exception("Conversation failed for prompt %s item %s", prompt.version, item.id)
            run_error = f"{type(e).__name__}: {e}"

        if run_error is None:
            try:
                grade = self.grader.grade(
                    question=item.question,
                    answer=answer,
                    ground_truth=item.ground_truth,
                    retrieved_context=retrieved_context if retrieved_context else None,
                )
            except Exception as e:
                logger.exception("Grading failed for prompt %s item %s", prompt.version, item.id)
                grade_error = f"{type(e).__name__}: {e}"

        error = run_error or grade_error
        transcript: dict[str, Any] = {
            "item_id": item.id,
            "category": item.category,
            "question": item.question,
            "ground_truth": item.ground_truth,
            "answer": answer,
            "retrieved_context": retrieved_context,
            "grade": grade.to_dict() if grade else None,
            "error": error,
            "prompt": f"{prompt.name}.{prompt.version}",
            "conversation_id": conv.conversation_id if conv else None,
            "messages": json.loads(
                json.dumps(conv.messages if conv else [], default=_serialize_block)
            ),
        }
        (prompt_dir / f"{item.id}.json").write_text(
            json.dumps(transcript, indent=2), encoding="utf-8"
        )

        return RunItemResult(
            item_id=item.id,
            prompt_version=f"{prompt.name}.{prompt.version}",
            question=item.question,
            ground_truth=item.ground_truth,
            answer=answer,
            grade=grade,
            error=error,
            category=item.category,
        )

    def _aggregate(self, results: list[RunItemResult]) -> list[PromptStats]:
        by_prompt: dict[str, list[RunItemResult]] = {}
        for r in results:
            by_prompt.setdefault(r.prompt_version, []).append(r)

        stats: list[PromptStats] = []
        for version, items in by_prompt.items():
            graded = [r for r in items if r.grade is not None]
            n_failures = len(items) - len(graded)
            if not graded:
                stats.append(
                    PromptStats(
                        prompt_version=version,
                        n_items=len(items),
                        n_failures=n_failures,
                        mean_overall=None,
                        mean_faithfulness=None,
                        mean_relevance=None,
                        mean_completeness=None,
                        mean_citation_quality=None,
                        mean_calibration=None,
                        failure_modes_counts={},
                    )
                )
                continue

            def mean(field: str) -> float:
                return sum(getattr(r.grade, field) for r in graded) / len(graded)

            fm_counts: dict[str, int] = {}
            for r in graded:
                for fm in r.grade.failure_modes:
                    fm_counts[fm] = fm_counts.get(fm, 0) + 1

            stats.append(
                PromptStats(
                    prompt_version=version,
                    n_items=len(items),
                    n_failures=n_failures,
                    mean_overall=mean("overall"),
                    mean_faithfulness=mean("faithfulness"),
                    mean_relevance=mean("relevance"),
                    mean_completeness=mean("completeness"),
                    mean_citation_quality=mean("citation_quality"),
                    mean_calibration=mean("calibration"),
                    failure_modes_counts=fm_counts,
                )
            )
        # Sort by mean_overall desc; None last.
        stats.sort(
            key=lambda s: (s.mean_overall is None, -(s.mean_overall or 0.0))
        )
        return stats

    def _format_leaderboard(
        self,
        stats: list[PromptStats],
        all_results: list[RunItemResult],
    ) -> str:
        lines = [
            f"# Leaderboard — {self.eval_dataset.name}",
            "",
            f"Items: {len(self.eval_dataset.items)} | Candidates: {len(stats)}",
            "",
            "| Rank | Prompt | N | Fails | Overall | Faith | Relev | Compl | Cite | Calib |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]

        def fmt(v: float | None) -> str:
            return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

        for i, s in enumerate(stats, 1):
            lines.append(
                f"| {i} | {s.prompt_version} | {s.n_items} | {s.n_failures} | "
                f"{fmt(s.mean_overall)} | {fmt(s.mean_faithfulness)} | "
                f"{fmt(s.mean_relevance)} | {fmt(s.mean_completeness)} | "
                f"{fmt(s.mean_citation_quality)} | {fmt(s.mean_calibration)} |"
            )

        # Per-prompt failure modes + worst examples.
        for s in stats:
            section = [f"\n## {s.prompt_version}"]
            if s.failure_modes_counts:
                section.append("\n**Failure modes (count):**")
                for fm, count in sorted(
                    s.failure_modes_counts.items(), key=lambda x: -x[1]
                ):
                    section.append(f"- `{fm}`: {count}")
            worst = [
                r for r in all_results
                if r.prompt_version == s.prompt_version and r.grade is not None
            ]
            worst.sort(key=lambda r: r.grade.overall)
            if worst:
                section.append("\n**Worst 3 by overall score:**")
                for r in worst[:3]:
                    section.append(
                        f"- `{r.item_id}` (overall {r.grade.overall:.2f}): "
                        f"{r.question[:120]}{'…' if len(r.question) > 120 else ''}"
                    )
            failures = [
                r for r in all_results
                if r.prompt_version == s.prompt_version and r.grade is None
            ]
            if failures:
                section.append("\n**Run/grade failures:**")
                for r in failures:
                    section.append(f"- `{r.item_id}`: {r.error}")
            lines.extend(section)

        return "\n".join(lines) + "\n"


# ---- helpers ----


def _serialize_block(obj: Any) -> Any:
    """JSON default for Anthropic SDK content blocks (pydantic models)."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"unserializable: {type(obj)}")


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_attr(block: Any, attr: str) -> Any:
    if isinstance(block, dict):
        return block.get(attr)
    return getattr(block, attr, None)


def _extract_retrieved_context(messages: list) -> list[str]:
    """Pull retrieve_from_papers tool_result content out of the message log."""
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if _block_type(block) == "tool_use":
                name = _block_attr(block, "name")
                tid = _block_attr(block, "id")
                if name and tid:
                    tool_id_to_name[tid] = name

    results: list[str] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if _block_type(block) != "tool_result":
                continue
            tid = _block_attr(block, "tool_use_id")
            if tool_id_to_name.get(tid) != "retrieve_from_papers":
                continue
            result_content = _block_attr(block, "content")
            if isinstance(result_content, str):
                results.append(result_content)
            elif isinstance(result_content, list):
                for sub in result_content:
                    text = _block_attr(sub, "text")
                    if text:
                        results.append(text)
    return results
