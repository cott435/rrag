"""Grader — LLM-as-judge scoring of QA responses against a rubric."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from anthropic import Anthropic
from anthropic.types import Message

from .config import DEFAULT_GRADING_MODEL
from .prompts import PromptRegistry, PromptTemplate

logger = logging.getLogger(__name__)


class GradingError(Exception):
    """Raised when the grader produces unparseable output even after retry."""


@dataclass
class GradeResult:
    faithfulness: int
    relevance: int
    completeness: int
    citation_quality: int
    calibration: int
    overall: float
    rationale: str
    failure_modes: list[str]

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "citation_quality": self.citation_quality,
            "calibration": self.calibration,
            "overall": self.overall,
            "rationale": self.rationale,
            "failure_modes": list(self.failure_modes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> GradeResult:
        # failure_modes is optional in the prompt's contract.
        return cls(
            faithfulness=int(data["faithfulness"]),
            relevance=int(data["relevance"]),
            completeness=int(data["completeness"]),
            citation_quality=int(data["citation_quality"]),
            calibration=int(data["calibration"]),
            overall=float(data["overall"]),
            rationale=str(data["rationale"]),
            failure_modes=list(data.get("failure_modes", [])),
        )


class Grader:
    """LLM-as-judge grader. Calls Claude with a structured-output rubric prompt.

    Temperature is pinned low (0.0) to reduce variance across runs.
    Use a different model from the one being graded when feasible to
    reduce self-preference bias (the project default for both is
    sonnet-4-5 — override `model` to break that tie when needed).
    """

    def __init__(
        self,
        client: Anthropic | None = None,
        model: str = DEFAULT_GRADING_MODEL,
        prompt: PromptTemplate | None = None,
        max_tokens: int | None = None,
        registry: PromptRegistry | None = None,
        temperature: float = 0.0,
    ):
        self._client = client or Anthropic()
        self.model = model
        self.prompt = prompt or (registry or PromptRegistry()).load("grading", "latest")
        self.max_tokens = max_tokens or int(self.prompt.metadata.get("max_tokens", 2000))
        self.temperature = temperature

    @property
    def prompt_version(self) -> str:
        return self.prompt.version

    def grade(
        self,
        question: str,
        answer: str,
        ground_truth: str | None = None,
        retrieved_context: list[str] | str | None = None,
    ) -> GradeResult:
        gt = ground_truth if ground_truth else "(not provided)"
        if retrieved_context is None:
            ctx = "(not provided)"
        elif isinstance(retrieved_context, list):
            ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(retrieved_context))
        else:
            ctx = retrieved_context

        system, user = self.prompt.render(
            question=question,
            answer=answer,
            ground_truth=gt,
            retrieved_context=ctx,
        )
        return self._call_with_retry(system, user)

    def _call_with_retry(self, system: str, user: str) -> GradeResult:
        messages: list[dict] = [{"role": "user", "content": user}]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            temperature=self.temperature,
        )
        text = _text_from_message(response)
        try:
            return _parse_grade(text)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Grader JSON parse failed; retrying once. %s", e)
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response wasn't valid JSON matching the schema. "
                        "Return only the JSON object — all required fields, no code "
                        "fences, no commentary."
                    ),
                }
            )
            response2 = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
                temperature=self.temperature,
            )
            text2 = _text_from_message(response2)
            try:
                return _parse_grade(text2)
            except (json.JSONDecodeError, KeyError, TypeError) as e2:
                raise GradingError(
                    f"Grader returned invalid JSON twice. "
                    f"Last response (truncated): {text2[:500]}"
                ) from e2


def _text_from_message(message: Message) -> str:
    return "\n".join(
        b.text for b in message.content if getattr(b, "type", None) == "text"
    )


def _parse_grade(text: str) -> GradeResult:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        nl = cleaned.find("\n")
        if nl != -1:
            cleaned = cleaned[nl + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
    return GradeResult.from_dict(json.loads(cleaned))
