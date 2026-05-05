"""Summarizer — calls Claude to produce a structured summary of a paper."""
from __future__ import annotations

import json
import logging

from anthropic import Anthropic
from anthropic.types import Message

from .config import DEFAULT_INFERENCE_MODEL
from .paper import Paper, StructuredSummary
from .prompts import PromptRegistry, PromptTemplate

logger = logging.getLogger(__name__)


class SummarizationError(Exception):
    """Raised when the LLM produces unparseable output even after retry."""


class Summarizer:
    """Calls Claude to produce a StructuredSummary for a Paper.

    Caches via Paper.set_summary, which writes under a key that includes
    the prompt version so different prompts don't poison each other's
    cached output (important for the prompt-optimization pipeline).
    """

    def __init__(
        self,
        client: Anthropic | None = None,
        model: str = DEFAULT_INFERENCE_MODEL,
        prompt: PromptTemplate | None = None,
        max_tokens: int | None = None,
        registry: PromptRegistry | None = None,
    ):
        self._client = client or Anthropic()
        self.model = model
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
        paper.set_summary(summary, prompt_version=self.prompt.version)
        return summary

    def _call_with_retry(self, system: str, user: str) -> StructuredSummary:
        """Call Claude once; on JSON parse failure, retry once with a follow-up."""
        messages: list[dict] = [{"role": "user", "content": user}]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        text = _text_from_message(response)
        try:
            return _parse_summary(text)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Summary JSON parse failed; retrying once. %s", e)
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
            )
            text2 = _text_from_message(response2)
            try:
                return _parse_summary(text2)
            except (json.JSONDecodeError, KeyError, TypeError) as e2:
                raise SummarizationError(
                    f"Summarizer returned invalid JSON twice. "
                    f"Last response (truncated): {text2[:500]}"
                ) from e2


def _text_from_message(message: Message) -> str:
    """Concatenate text blocks from an Anthropic Message."""
    return "\n".join(
        b.text for b in message.content if getattr(b, "type", None) == "text"
    )


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
