"""Interactive Q&A REPL over the paper corpus, with optional web_search."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.research_rag import Embedder, LLMClient, PromptTemplate
from src.research_rag.config import (
    DEFAULT_INFERENCE_MODEL,
    PAPERS_DIR,
    PROMPTS_DIR,
    ensure_dirs,
)
from src.research_rag.conversation import Conversation
from src.research_rag.corpus import PaperCorpus
from src.research_rag.tools import make_corpus_tools, web_search_tool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive Q&A over the paper corpus, with optional web search."
    )
    parser.add_argument("--resume", type=str, help="Conversation id to resume.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="qa.v1",
        help="Prompt as <task>.<version> (default: qa.v1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_INFERENCE_MODEL,
        help=(
            f"Model id (default: {DEFAULT_INFERENCE_MODEL}). "
            "Use a 'claude-*' id for Anthropic; anything else (e.g. 'qwen3:8b') "
            "routes through local Ollama and disables tool use."
        ),
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable corpus tools entirely (required for non-Claude models).",
    )
    parser.add_argument("--papers-dir", type=Path, default=PAPERS_DIR)
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable the hosted web_search tool.",
    )
    parser.add_argument(
        "--web-search-domains",
        type=str,
        default=None,
        help="Comma-separated allowed domains for web_search.",
    )
    parser.add_argument(
        "--web-search-max-uses",
        type=int,
        default=5,
        help="Maximum web_search invocations per turn (default: 5).",
    )
    parser.add_argument(
        "--show-tool-calls",
        action="store_true",
        help="Print tool calls and (truncated) results as they happen.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ensure_dirs()

    if "." not in args.prompt:
        print(f"--prompt must be <task>.<version> (e.g. qa.v1), got '{args.prompt}'", file=sys.stderr)
        return 1
    task, version = args.prompt.rsplit(".", 1)
    prompt_path = PROMPTS_DIR / task / f"{version}.md"
    if not prompt_path.exists():
        print(f"Prompt not found: {prompt_path}", file=sys.stderr)
        return 1
    prompt = PromptTemplate.from_file(prompt_path)

    embedder = Embedder()
    corpus = PaperCorpus(embedder=embedder, papers_dir=args.papers_dir)
    corpus.discover()

    if args.no_tools:
        tools = []
    else:
        tools = make_corpus_tools(corpus)
        if not args.no_web_search:
            domains = (
                [d.strip() for d in args.web_search_domains.split(",") if d.strip()]
                if args.web_search_domains
                else None
            )
            tools.append(
                web_search_tool(
                    max_uses=args.web_search_max_uses,
                    allowed_domains=domains,
                )
            )

    on_tool_call = _print_tool_call if args.show_tool_calls else None

    llm = LLMClient(model=args.model)
    conv = Conversation(
        llm=llm,
        corpus=corpus,
        system_prompt=prompt.system,
        tools=tools,
        conversation_id=args.resume,
        on_tool_call=on_tool_call,
    )

    summarized = sum(1 for p in corpus.papers.values() if p.has_summary())
    print(
        f"Conversation {conv.conversation_id} | model={args.model} | "
        f"prompt={prompt.name}.{prompt.version}"
    )
    print(
        f"Corpus: {len(corpus.papers)} papers ({summarized} summarized) | "
        f"Tools: {[t.name for t in tools]}"
    )
    print("Type 'exit' (or Ctrl-D) to quit.\n")

    while True:
        try:
            line = input("You: ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        try:
            answer = conv.ask(line)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        print(f"\nAssistant: {answer}\n")

    print(f"Saved conversation: {conv.conversation_id}")
    return 0


def _print_tool_call(name: str, inputs: dict, result: str) -> None:
    snippet = result if len(result) <= 200 else result[:200] + "..."
    snippet = snippet.replace("\n", " ")
    print(f"  [tool] {name}({inputs}) -> {snippet}")


if __name__ == "__main__":
    sys.exit(main())
