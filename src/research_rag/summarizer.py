"""Summarizer — calls an LLM to produce a structured summary of a paper."""
from __future__ import annotations

import json
import logging

from .llm import LLMClient
from .paper import Paper, StructuredSummary
from .prompts import PromptRegistry, PromptTemplate

logger = logging.getLogger(__name__)


class SummarizationError(Exception):
    """Raised when the LLM produces unparseable output even after retry."""


class Summarizer:
    """Produces a StructuredSummary for a Paper via an :class:`LLMClient`.

    Caches via Paper.set_summary, which writes under a key that includes
    the prompt version so different prompts don't poison each other's
    cached output (important for the prompt-optimization pipeline).
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        prompt: PromptTemplate | None = None,
        max_tokens: int | None = None,
        registry: PromptRegistry | None = None,
    ):
        self.llm = llm or LLMClient(model=_default_inference_model())
        self.prompt = prompt or (registry or PromptRegistry()).load("summarization", "latest")
        # Honor the prompt's declared max_tokens unless explicitly overridden.
        self.max_tokens = max_tokens or int(self.prompt.metadata.get("max_tokens", 4000))

    @property
    def prompt_version(self) -> str:
        return self.prompt.version

    def summarize(self, paper: Paper, force: bool = False) -> StructuredSummary:
        """Return the paper's structured summary, computing if missing."""
        if not force and paper.has_summary(prompt_version=self.prompt.version):
            try:
                return paper.get_summary(prompt_version=self.prompt.version)
            except ValueError:
                pass  # corrupt cache was already cleared by get_summary; recompute

        text = paper.text
        # Long-paper map-reduce is a known follow-up; for now rely on the
        # 200k context window of Sonnet 4.5 / Haiku 4.5 to fit the paper.
        system, user = self.prompt.render(paper_text=text)
        summary = self._call_with_retry(system, user)
        paper.set_summary(summary, self.llm.model, prompt_version=self.prompt.version)
        return summary

    def _call_with_retry(self, system: str, user: str) -> StructuredSummary:
        """Call the LLM once; on JSON parse failure, retry once with a follow-up."""
        messages: list[dict] = [{"role": "user", "content": user}]
        resp = self.llm.chat(
            messages=messages,
            system=system,
            tools=None,
            max_tokens=self.max_tokens,
        )
        try:
            return _parse_summary(resp.text)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Summary JSON parse failed; retrying once. %s", e)
            messages.append({"role": "assistant", "content": resp.text})
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
            resp2 = self.llm.chat(
                messages=messages,
                system=system,
                tools=None,
                max_tokens=self.max_tokens,
            )
            try:
                return _parse_summary(resp2.text)
            except (json.JSONDecodeError, KeyError, TypeError) as e2:
                raise SummarizationError(
                    f"Summarizer returned invalid JSON twice. "
                    f"Last response (truncated): {resp2.text[:500]}"
                ) from e2


def _default_inference_model() -> str:
    # Imported lazily so tests that monkeypatch config defaults work.
    from .config import DEFAULT_INFERENCE_MODEL

    return DEFAULT_INFERENCE_MODEL


def _parse_summary(text: str) -> StructuredSummary:
    """Parse a JSON response into a StructuredSummary, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
    data = json.loads(cleaned)
    return StructuredSummary.from_dict(data)
