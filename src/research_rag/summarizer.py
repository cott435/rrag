"""Summarizer — calls an LLM to produce a structured summary of a paper.

Two summarization paths:

* **Single-shot JSON** (the default, used for Claude): one call asks the model
  to produce the full :class:`StructuredSummary` as a JSON object.
* **Chained per-field** (the fallback for local/Ollama models): one call per
  field, each returning plain text. The Summarizer assembles the
  StructuredSummary itself. This is used because weak local models (e.g.
  ``qwen3:4b``) tend to ignore the JSON schema and emit markdown prose.

The chained path is engaged when the single-shot attempt fails to parse AND
the provider is Ollama. Claude keeps its original "fix your JSON" retry.
"""
from __future__ import annotations

import json
import logging
import re

from .llm import LLMClient, LLMProvider
from .paper import Paper, StructuredSummary
from .prompts import PromptRegistry, PromptTemplate

logger = logging.getLogger(__name__)


class SummarizationError(Exception):
    """Raised when the LLM produces unparseable output even after retry."""


# Per-field instructions used by the chained fallback path.
# Mirrors the single-shot JSON schema's field-by-field guidance so the
# resulting StructuredSummary is comparable across paths.
_FIELD_INSTRUCTIONS: list[dict] = [
    {
        "name": "tldr",
        "kind": "string",
        "instructions": (
            "Write 1-2 sentences capturing the single most important takeaway: "
            "what the paper did and what it found. Written so a researcher in "
            "an adjacent field could understand it. No preamble."
        ),
    },
    {
        "name": "problem",
        "kind": "string",
        "instructions": (
            "Write 2-4 sentences describing the problem or question that "
            "motivates this work and the gap in prior work it addresses."
        ),
    },
    {
        "name": "method",
        "kind": "string",
        "instructions": (
            "Write 3-6 sentences describing how the authors approached the "
            "problem. Name specific techniques, datasets, model architectures, "
            "or experimental setups using their standard names when available."
        ),
    },
    {
        "name": "key_results",
        "kind": "string",
        "instructions": (
            "Write 3-6 sentences describing the findings. Include specific "
            "numbers, comparisons, and qualitative findings where the paper "
            "reports them. Distinguish headline results from secondary ones."
        ),
    },
    {
        "name": "limitations",
        "kind": "string",
        "instructions": (
            "Write 2-4 sentences describing limitations: cases where the "
            "method fails, datasets it wasn't tested on, or assumptions that "
            "may not hold. If the paper does not discuss limitations, reply "
            "exactly: Not explicitly discussed."
        ),
    },
    {
        "name": "contributions",
        "kind": "list",
        "instructions": (
            "List 2-5 distinct contributions claimed by the paper. Output ONE "
            "contribution per line. Do not number the lines. Do not add a "
            "leading bullet character. Each line is a complete sentence."
        ),
    },
    {
        "name": "keywords",
        "kind": "list",
        "instructions": (
            "List 4-8 technical terms, methods, and concepts that characterize "
            "this paper. Output ONE keyword or short phrase per line. No "
            "numbering, no bullets, no explanations — just the term itself."
        ),
    },
]


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
        chain_prompt: PromptTemplate | None = None,
    ):
        self.llm = llm or LLMClient(model=_default_inference_model())
        self._registry = registry or PromptRegistry()
        self.prompt = prompt or self._registry.load("summarization", "latest")
        # Honor the prompt's declared max_tokens unless explicitly overridden.
        self.max_tokens = max_tokens or int(self.prompt.metadata.get("max_tokens", 4000))
        # Lazily loaded the first time the chained fallback fires, so a missing
        # chain prompt doesn't break Anthropic-only deployments.
        self._chain_prompt: PromptTemplate | None = chain_prompt

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
        summary = self._call_with_retry(system, user, paper_text=text)
        paper.set_summary(summary, self.llm.model, prompt_version=self.prompt.version)
        return summary

    def _call_with_retry(
        self,
        system: str,
        user: str,
        *,
        paper_text: str,
    ) -> StructuredSummary:
        """Call the LLM once; on JSON parse failure, retry strategy depends on provider.

        * Anthropic: a second call with a corrective "your previous response
          wasn't valid JSON" follow-up.
        * Ollama: switch to the per-field chained path, which is much more
          forgiving with weak local models.
        """
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
            logger.warning("Summary JSON parse failed; falling back to retry path. %s", e)
            if self.llm.provider is LLMProvider.OLLAMA:
                logger.info(
                    "Local model (%s) — switching to chained per-field summarization.",
                    self.llm.model,
                )
                return self._chained_summarize(paper_text)

            # Anthropic — corrective JSON retry.
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

    def _chained_summarize(self, paper_text: str) -> StructuredSummary:
        """Build a StructuredSummary by calling the model once per field.

        Used as a fallback when the single-shot JSON path fails on a local
        model. Each field gets its own focused prompt and a small max_tokens
        budget — much easier for a 4B-parameter model to handle than the
        full schema in one shot.
        """
        chain_prompt = self._get_chain_prompt()
        per_field_max_tokens = int(chain_prompt.metadata.get("max_tokens", 1000))

        fields: dict[str, str | list[str]] = {}
        for spec in _FIELD_INSTRUCTIONS:
            system, user = chain_prompt.render(
                field_name=spec["name"],
                field_instructions=spec["instructions"],
                paper_text=paper_text,
            )
            try:
                resp = self.llm.chat(
                    messages=[{"role": "user", "content": user}],
                    system=system,
                    tools=None,
                    max_tokens=per_field_max_tokens,
                )
            except Exception as e:  # noqa: BLE001 — surface as SummarizationError
                raise SummarizationError(
                    f"Chained summarization failed on field {spec['name']!r}: {e}"
                ) from e

            if spec["kind"] == "list":
                fields[spec["name"]] = _parse_list_field(resp.text)
            else:
                fields[spec["name"]] = _parse_string_field(resp.text)

        # StructuredSummary.from_dict requires every field — _FIELD_INSTRUCTIONS
        # covers all of them, so this should never KeyError.
        return StructuredSummary.from_dict(fields)

    def _get_chain_prompt(self) -> PromptTemplate:
        if self._chain_prompt is None:
            self._chain_prompt = self._registry.load("summarization_chain", "latest")
        return self._chain_prompt


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


# ---- chained-path response parsing ----

# Lines like "- foo", "* foo", "1. foo", "2) foo", "• foo".
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)")
# Markdown header lines we want to skip when the model adds them anyway.
_MD_HEADER_RE = re.compile(r"^\s*#{1,6}\s")


def _strip_code_fences(text: str) -> str:
    """Remove surrounding ``` fences if the model wrapped its answer."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        nl = cleaned.find("\n")
        if nl != -1:
            cleaned = cleaned[nl + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def _parse_string_field(text: str) -> str:
    """Normalize a free-text field response.

    Strips code fences and leading markdown headers the model may have
    emitted despite instructions. Returns the cleaned text, or
    "Not explicitly discussed." if the model produced effectively nothing.
    """
    cleaned = _strip_code_fences(text)
    # Drop a single leading markdown header line if present.
    lines = cleaned.splitlines()
    while lines and _MD_HEADER_RE.match(lines[0]):
        lines.pop(0)
    cleaned = "\n".join(lines).strip()
    return cleaned or "Not explicitly discussed."


def _parse_list_field(text: str) -> list[str]:
    """Parse a list-field response into a list of strings.

    Accepts any combination of bullet styles ("- foo", "* foo", "• foo"),
    numbered lists ("1. foo", "2) foo"), or one-per-line plain text. Skips
    blank lines and markdown headers.
    """
    cleaned = _strip_code_fences(text)
    items: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _MD_HEADER_RE.match(line):
            continue
        line = _LIST_PREFIX_RE.sub("", line).strip()
        if line:
            items.append(line)
    return items
