"""Tool schemas + dispatchers for the Conversation tool loop.

Each local tool is a (schema, handler) pair. The Conversation maintains
a name -> handler map and dispatches on tool_use blocks. Hosted tools
(handler=None) — currently web_search — are executed server-side by
Anthropic; we only need to include the schema in the tools list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .corpus import PaperCorpus


@dataclass
class Tool:
    name: str
    schema: dict[str, Any]
    handler: Callable[..., Any] | None  # None for hosted tools (e.g. web_search)


def make_corpus_tools(corpus: PaperCorpus) -> list[Tool]:
    """Build the corpus-backed tools: retrieve, read_summary, read_section, list_papers."""

    def retrieve_from_papers(query: str, k: int = 8) -> str:
        results = corpus.search_chunks(query=query, k_chunks=k)
        if not results:
            return "No matching chunks found in the corpus."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.paper_title or r.paper_id
            section = r.section or "(no section)"
            lines.append(
                f"[{i}] paper_id={r.paper_id} | title={title} | section={section} | "
                f"chunk_index={r.chunk_index} | score={r.score:.4f}\n"
                f"{r.content}"
            )
        return "\n\n".join(lines)

    def read_paper_summary(paper_id: str) -> str:
        try:
            paper = corpus.get(paper_id)
        except KeyError:
            return (
                f"No paper with id {paper_id} in corpus. "
                f"Use list_papers to see available ids."
            )
        if not paper.has_summary():
            return (
                f"No summary cached for {paper_id}. "
                f"Run scripts/summarize.py --paper {paper_id} to generate it."
            )
        s = paper.summary
        contributions = "\n".join(f"- {c}" for c in s.contributions)
        return (
            f"# {paper.title or paper.paper_id}\n\n"
            f"**TL;DR:** {s.tldr}\n\n"
            f"**Problem:** {s.problem}\n\n"
            f"**Method:** {s.method}\n\n"
            f"**Key results:** {s.key_results}\n\n"
            f"**Limitations:** {s.limitations}\n\n"
            f"**Contributions:**\n{contributions}\n\n"
            f"**Keywords:** {', '.join(s.keywords)}"
        )

    def read_paper_section(
        paper_id: str,
        section_name: str | None = None,
        chunk_index: int | None = None,
    ) -> str:
        try:
            paper = corpus.get(paper_id)
        except KeyError:
            return f"No paper with id {paper_id} in corpus."
        chunks = paper.chunks
        if chunk_index is not None:
            for c in chunks:
                if c.chunk_index == chunk_index:
                    section = c.section or "(no section)"
                    return f"## {section}\n\n{c.content}"
            return (
                f"No chunk with index {chunk_index} in paper {paper_id} "
                f"(total chunks: {len(chunks)})."
            )
        if section_name:
            needle = section_name.lower()
            matches = [c for c in chunks if c.section and needle in c.section.lower()]
            if not matches:
                return f"No section matching '{section_name}' in paper {paper_id}."
            return "\n\n".join(f"## {c.section}\n\n{c.content}" for c in matches)
        return "Provide either section_name or chunk_index."

    def list_papers() -> str:
        if not corpus.papers:
            return "No papers in corpus."
        lines = []
        for paper in corpus.papers.values():
            authors = ", ".join(paper.authors) if paper.authors else "?"
            year = paper.year or "?"
            title = paper.title or "(untitled)"
            mark = "+" if paper.has_summary() else "-"
            lines.append(f"[{mark}] {paper.paper_id}  {title}  ({authors}, {year})")
        return "Available papers ('+' = has summary):\n" + "\n".join(lines)

    return [
        Tool(
            name="retrieve_from_papers",
            handler=retrieve_from_papers,
            schema={
                "name": "retrieve_from_papers",
                "description": (
                    "Search the paper corpus for chunks relevant to a query. "
                    "Two-tier retrieval: paper summaries first, then chunks within "
                    "the most relevant papers. Hybrid (BM25 + dense) fused with RRF. "
                    "Returns ranked chunks with paper_id, section, chunk_index, and score."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "k": {
                            "type": "integer",
                            "description": "Maximum chunks to return.",
                            "default": 8,
                        },
                    },
                    "required": ["query"],
                },
            },
        ),
        Tool(
            name="read_paper_summary",
            handler=read_paper_summary,
            schema={
                "name": "read_paper_summary",
                "description": (
                    "Read the cached structured summary for a paper (tldr, problem, "
                    "method, key_results, limitations, contributions, keywords). "
                    "Cheap; use this before reasoning in detail about a paper to "
                    "orient yourself before drilling into chunks."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "paper_id": {
                            "type": "string",
                            "description": "Paper id (16-char hex).",
                        },
                    },
                    "required": ["paper_id"],
                },
            },
        ),
        Tool(
            name="read_paper_section",
            handler=read_paper_section,
            schema={
                "name": "read_paper_section",
                "description": (
                    "Read the full text of a section, by section name "
                    "(case-insensitive substring) or by chunk_index from a "
                    "previous retrieve_from_papers result. Use when a retrieved "
                    "chunk needs more surrounding context."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        "section_name": {
                            "type": "string",
                            "description": "Section name (substring match).",
                        },
                        "chunk_index": {
                            "type": "integer",
                            "description": "Specific chunk to read.",
                        },
                    },
                    "required": ["paper_id"],
                },
            },
        ),
        Tool(
            name="list_papers",
            handler=list_papers,
            schema={
                "name": "list_papers",
                "description": (
                    "List every paper in the corpus with id, title, authors, year, "
                    "and whether a summary is cached."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
        ),
    ]


def web_search_tool(
    max_uses: int = 5,
    allowed_domains: list[str] | None = None,
) -> Tool:
    """Anthropic's hosted web_search tool. Executed server-side; no local handler."""
    schema: dict[str, Any] = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }
    if allowed_domains:
        schema["allowed_domains"] = allowed_domains
    return Tool(name="web_search", schema=schema, handler=None)
