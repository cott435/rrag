"""Tests for EvalDataset loading and PromptOptimizationPipeline.run()."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from research_rag import (
    EvalDataset,
    GradeResult,
    PromptOptimizationPipeline,
    PromptTemplate,
)


# ---- EvalDataset ----


def test_evaldataset_from_file(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(
        json.dumps(
            {
                "name": "demo",
                "items": [
                    {"id": "q1", "question": "What?", "ground_truth": "It.", "category": "factual"},
                    {"id": "q2", "question": "Why?", "relevant_paper_ids": ["x", "y"]},
                ],
            }
        )
    )
    ds = EvalDataset.from_file(p)
    assert ds.name == "demo" and len(ds.items) == 2
    assert ds.items[0].id == "q1" and ds.items[0].category == "factual"
    assert ds.items[1].relevant_paper_ids == ["x", "y"]


def test_evaldataset_handles_missing_optional_fields(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps({"name": "minimal", "items": [{"id": "q1", "question": "?"}]}))
    ds = EvalDataset.from_file(p)
    assert ds.items[0].ground_truth is None
    assert ds.items[0].relevant_paper_ids == []


# ---- PromptOptimizationPipeline ----


class _StubConv:
    def __init__(self, answer):
        self.conversation_id = "stub"
        self._answer = answer
        self.messages: list = []

    def ask(self, q):
        self.messages.append({"role": "user", "content": q})
        self.messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": self._answer}]}
        )
        return self._answer


def _gr(overall, **kw):
    base = dict(
        faithfulness=4, relevance=4, completeness=4,
        citation_quality=4, calibration=4, overall=overall,
        rationale="r", failure_modes=[],
    )
    base.update(kw)
    return GradeResult(**base)


def test_pipeline_writes_run_dir_and_ranks_prompts(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    eval_path = tmp_path / "ds.json"
    eval_path.write_text(json.dumps({
        "name": "tiny",
        "items": [{"id": "q1", "question": "Q?", "ground_truth": "GT"}],
    }))
    dataset = EvalDataset.from_file(eval_path)

    p1 = PromptTemplate(name="qa", version="v1", system="s", user_template="", metadata={"name": "qa", "version": "v1"})
    p2 = PromptTemplate(name="qa", version="v2", system="s2", user_template="", metadata={"name": "qa", "version": "v2"})

    grades = iter([_gr(3.0), _gr(5.0)])
    grader = MagicMock(); grader.grade = lambda **kw: next(grades)

    answers = {"v1": "answer 1", "v2": "answer 2"}
    pipeline = PromptOptimizationPipeline(
        corpus=MagicMock(),
        eval_dataset=dataset,
        grader=grader,
        candidate_prompts=[p1, p2],
        runner_factory=lambda prompt: _StubConv(answers[prompt.version]),
        runs_dir=runs_dir,
    )
    result = pipeline.run()

    assert (result.run_dir / "leaderboard.md").exists()
    assert (result.run_dir / "summary.json").exists()
    assert (result.run_dir / "qa.v1" / "q1.json").exists()
    assert (result.run_dir / "qa.v2" / "q1.json").exists()
    # v2 (overall=5) should outrank v1 (overall=3).
    assert [s.prompt_version for s in result.stats] == ["qa.v2", "qa.v1"]


def test_pipeline_records_run_failures(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    eval_path = tmp_path / "ds.json"
    eval_path.write_text(json.dumps({"name": "x", "items": [{"id": "q1", "question": "?"}]}))
    dataset = EvalDataset.from_file(eval_path)

    class _CrashConv:
        conversation_id = "c"
        messages: list = []

        def ask(self, q):
            raise RuntimeError("conv crashed")

    p = PromptTemplate(name="qa", version="v1", system="s", user_template="", metadata={"name": "qa", "version": "v1"})
    grader = MagicMock()
    pipeline = PromptOptimizationPipeline(
        corpus=MagicMock(),
        eval_dataset=dataset,
        grader=grader,
        candidate_prompts=[p],
        runner_factory=lambda _: _CrashConv(),
        runs_dir=runs_dir,
    )
    result = pipeline.run()
    assert result.stats[0].n_failures == 1
    grader.grade.assert_not_called()


def test_pipeline_extracts_retrieved_context_for_grader(tmp_path):
    """Grader should receive retrieve_from_papers tool_results as context."""
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    eval_path = tmp_path / "ds.json"
    eval_path.write_text(json.dumps({
        "name": "ctx", "items": [{"id": "q1", "question": "Q?"}],
    }))
    dataset = EvalDataset.from_file(eval_path)

    class _RetrieveConv:
        conversation_id = "c"

        def __init__(self):
            self.messages = []

        def ask(self, q):
            self.messages = [
                {"role": "user", "content": q},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "u1", "name": "retrieve_from_papers", "input": {"query": q}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "u1", "content": "CHUNK CONTENT"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
            return "answer"

    p = PromptTemplate(name="qa", version="v1", system="s", user_template="", metadata={"name": "qa", "version": "v1"})
    captured: list = []
    grader = MagicMock()
    grader.grade = lambda **kw: (captured.append(kw), _gr(4.0))[1]

    pipeline = PromptOptimizationPipeline(
        corpus=MagicMock(),
        eval_dataset=dataset,
        grader=grader,
        candidate_prompts=[p],
        runner_factory=lambda _: _RetrieveConv(),
        runs_dir=runs_dir,
    )
    pipeline.run()

    assert captured[0]["retrieved_context"] == ["CHUNK CONTENT"]
