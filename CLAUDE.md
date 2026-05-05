# CLAUDE.md

Conventions and guidance for Claude Code while working in this repository. Read this before making changes.

---

## Project Summary

A local RAG system specialized in scientific research papers. Two main capabilities: paper summarization and research Q&A over a paper corpus with web search supplementation. Includes a prompt optimization pipeline with an LLM-as-judge grader.

See `PROJECT_PLAN.md` for the full design. This file covers how to work in the codebase.

---

## Tech Stack (do not substitute without asking)

- **Python 3.10+**
- **`anthropic`** SDK for LLM calls. Default models: `claude-sonnet-4-5` for inference/grading, `claude-haiku-4-5` for cheap bulk operations.
- **`voyageai`** for embeddings. Model: `voyage-3-large`.
- **`pypdf`** for PDF parsing, `pdfplumber` as fallback, `pytesseract` only behind an `--ocr` flag.
- **`pytest`** for tests.
- **`python-dotenv`** for env loading.
- **SQLite** (stdlib `sqlite3`) for conversation history. **Pickle** for index serialization. **JSON** for everything human-readable (summaries, eval datasets, configs).

No external vector DB, no LangChain, no LlamaIndex. The retrieval stack is hand-rolled — that's intentional.

---

## Code Style

- **Type hints everywhere.** Function signatures and dataclass fields. Don't bother annotating obvious local variables.
- **Dataclasses over dicts** for any structure that crosses a function boundary more than twice. `StructuredSummary`, `ChunkResult`, `GradeResult`, etc. are dataclasses.
- **Pathlib over `os.path`.** Always.
- **f-strings over `.format()` or `%`.**
- **Explicit imports.** No `from x import *`.
- **Docstrings on public methods only.** Private helpers (leading underscore) don't need them unless the logic is non-obvious. One-line docstrings are fine.
- **Line length 100.** Configure ruff/black accordingly.
- **No silent `except Exception:`.** Catch specific exceptions. If you genuinely need a broad catch, log the exception and re-raise or include a comment explaining why.

---

## Architectural Rules

### Lazy by default
Anything expensive (LLM calls, embeddings, parsing) checks cache before computing. The `Paper` class properties are the model — see `paper.py`. New expensive operations follow the same pattern: hash-keyed cache file, check on read, write on first compute.

### Cache keys must include prompt version where relevant
Summaries depend on the summarization prompt; their cache keys include the prompt version. Same for any other artifact whose content depends on a prompt. Without this, prompt optimization runs become meaningless. **Don't shortcut this.**

### Classes own their state; scripts wire them
`scripts/` contains thin entry points only — argparse, config loading, instantiating classes, calling methods. Business logic lives in `src/research_rag/`. If a script grows past ~100 lines, something belongs in a class.

### One Retriever interface
`VectorIndex`, `BM25Index`, and `Retriever` follow the protocol from `chunking_embed.ipynb`. New index types implement the same `add_document` / `add_documents` / `search` contract. The `Retriever` fuses with RRF — don't reimplement fusion elsewhere.

### Tools are data + dispatcher pairs
Each tool in `tools.py` is (a) a JSON schema constant and (b) a Python function. The `Conversation` class accepts a list of tools and dispatches by name in its loop. Don't hardcode tool names inside `Conversation`.

---

## Caching & File Layout Discipline

- **Never write to `papers/` from code.** That directory is user-owned input.
- **All generated artifacts go under `cache/`.** Mirror the structure in `PROJECT_PLAN.md`.
- **Conversations go to `conversations/` (SQLite).** Don't store conversation history in JSON files.
- **Eval runs are immutable once written.** A run goes to `eval/runs/{ISO_TIMESTAMP}/` and isn't modified afterward. If a run needs to be re-graded, write a new run.
- **Prompts are markdown with YAML frontmatter under `prompts/`.** Versioned by filename (`qa.v1.md`, `qa.v2.md`). The registry reads filesystem state — don't add a separate prompt index file.

---

## LLM Call Conventions

- **Use the helper-function pattern from `using_tools.ipynb`** for low-level chat: `add_user_message`, `add_assistant_message`, `chat`, `text_from_message`. Keep these in `conversation.py` as private helpers; the `Conversation` class wraps them.
- **Always pass `max_tokens` explicitly.** Default to 1000 for short responses, 4000 for summarization, 2000 for grading. No magic defaults.
- **Tool loop terminates on `stop_reason != "tool_use"`.** Bound iterations with a max (default 10) to prevent runaway loops in tests.
- **Web search uses `web_search_20250305`** with a configurable `max_uses` (default 5) and optional `allowed_domains`. See `using_web_search.ipynb` for the schema shape.
- **Structured outputs from LLMs are JSON-in-text, not tool calls.** For `Summarizer` and `Grader`, prompt for JSON, parse with `json.loads`, retry once on `JSONDecodeError` with a "your previous response wasn't valid JSON, try again" follow-up. Don't use tool-use to coerce structure — it's overkill here.

---

## Testing

- **Mock LLM and embedding calls in unit tests.** Tests must run offline and finish in under 30 seconds total.
- **Use real fixtures for parsing and indexing.** Put 1–2 small PDFs in `tests/fixtures/papers/` (public-domain or arXiv preprints). These exercise the ingestion path without LLM calls.
- **Cache tests get their own `tmp_path`.** Never let tests touch the real `cache/` directory.
- **Smoke test for the conversation loop** with a mocked client that returns a canned `tool_use` then a canned text reply. Verifies the loop dispatches correctly.
- **Don't test prompt content.** Prompts change; their effects are measured via the eval pipeline, not pytest.

---

## Error Handling Patterns

- **Voyage rate limits:** the embedder retries with exponential backoff (start 1s, max 32s, max 5 retries). The user's notebook noted this — don't lose it.
- **Anthropic API errors:** propagate by default. Only catch and retry on `RateLimitError` and `APITimeoutError`.
- **PDF parsing failures:** try `pypdf` → `pdfplumber` → raise a clear `PaperIngestionError` that names the file. Don't fall through to OCR silently; require the explicit flag.
- **Cache corruption:** if a cache file fails to load (pickle error, malformed JSON), log a warning, delete it, and recompute. Don't crash on bad cache.

---

## What NOT to Do

- Don't add a vector database dependency (Chroma, Qdrant, Weaviate, etc.). Local pickle is the design.
- Don't replace the hand-rolled retriever with a library. The user wrote it; mirror their code.
- Don't auto-summarize all papers on ingestion unless `--summarize` is passed. Summarization is lazy.
- Don't store API keys anywhere except `.env`. `.env` is gitignored. `.env.example` is committed and contains placeholder values only.
- Don't write Markdown lists for things that aren't lists. Prose is fine in docs.
- Don't add a web UI. CLI scripts only for v1.

---

## When Adding a New Feature

1. Update `PROJECT_PLAN.md` if the feature changes architecture.
2. Add the class or function in `src/research_rag/`.
3. Add a thin script under `scripts/` if it's user-facing.
4. Add tests under `tests/` (with mocks for LLM/embedding).
5. If it introduces new prompts, version them and add to `prompts/`.
6. If it introduces new cached artifacts, document the cache key and location in `PROJECT_PLAN.md`'s caching table.

---

## Open Questions / Known Gaps

These are flagged in the plan and should not be silently resolved without checking with the user:

- **Section chunking fallback** — the markdown `## ` split fails on poorly-converted PDFs. Plan calls for a fixed-window fallback; implement only when first paper triggers it.
- **Cross-encoder reranking** — interface scaffolded but not implemented in v1.
- **Long-conversation summarization** — defer until context overflow is observed in real use.

---

## Useful Commands

```bash
# First-time setup
cp .env.example .env  # then fill in keys
pip install -e .

# Ingest all PDFs in papers/
python -m scripts.ingest_papers

# Summarize one paper (lazy, uses cache)
python -m scripts.summarize --paper <paper_id>

# Start a Q&A session
python -m scripts.ask

# Run prompt optimization
python -m scripts.optimize_prompts --task qa --candidates qa.v1,qa.v2 --eval-set eval/datasets/research_qa.json

# Run tests
pytest -xvs
```
