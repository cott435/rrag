# Research RAG System — Planning Document

## Project Overview

Build a RAG-based research assistant specialized in scientific papers. The system has two primary user-facing capabilities:

1. **Paper Summarization** — generate (and cache) summaries of individual papers on demand.
2. **Research Inference / Q&A** — answer questions using a corpus of papers as the primary knowledge source, supplemented by web search when paper coverage is insufficient.

A secondary capability is a **prompt engineering pipeline** with a grader so the user can iterate on prompts and measure quality across a held-out set of papers and questions.

### Design Principles

- **Lazy by default.** Summaries, embeddings, and BM25 indexes are computed once and cached to disk. On every run, check cache first; only compute what's missing. Cache invalidation keys on a content hash of the source paper.
- **Class-based, composable architecture.** Conversations and retrieval components are objects that own their state. Scripts are thin entry points that wire classes together.
- **Two-tier retrieval.** Both paper-level summary embeddings and chunk-level embeddings are indexed. Retrieval is summary-first, then chunks within selected papers (see Architecture).
- **Hybrid search.** Every retrieval layer runs both dense (Voyage) and BM25, fused with Reciprocal Rank Fusion — same pattern as the user's existing `chunking_embed.ipynb`.
- **Web search is a tool, not a fallback.** Claude decides when to invoke it via tool use, with citations preserved.

---

## Tech Stack

- **Language:** Python 3.10+
- **LLM:** Anthropic Claude via the `anthropic` SDK. Default to `claude-sonnet-4-5` for inference and grading; `claude-haiku-4-5` for cheaper bulk operations like summarization of many papers (configurable per-class).
- **Embeddings:** VoyageAI (`voyage-3-large`), matching the user's existing notebook.
- **PDF parsing:** `pypdf` for text extraction; fall back to `pdfplumber` for papers where `pypdf` produces garbage. Provide a `--ocr` flag that uses `pytesseract` for scanned papers (off by default).
- **Storage:** Local filesystem only. JSON for metadata and summaries, pickle for embedding indexes, SQLite for conversation history. No external DB.
- **Env:** `python-dotenv` for `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY`.
- **Testing:** `pytest`. Mock LLM/embedding calls.

---

## Repository Layout

```
research_rag/
├── papers/                       # User drops PDFs here
├── cache/
│   ├── parsed/                   # {paper_id}.md  — extracted text
│   ├── summaries/                # {paper_id}.json — structured summary
│   ├── embeddings/               # {paper_id}.pkl — chunk embeddings
│   └── bm25/                     # {paper_id}.pkl — per-paper BM25 state
│   └── corpus_index/             # corpus-level summary index
├── conversations/                # SQLite DB of saved chats
├── prompts/
│   ├── summarization/            # versioned prompt templates
│   ├── qa/
│   └── grading/
├── eval/
│   ├── datasets/                 # JSON test sets (questions + ground truth)
│   └── runs/                     # timestamped eval results
├── src/research_rag/
│   ├── __init__.py
│   ├── config.py                 # paths, model names, defaults
│   ├── ingest.py                 # PDF → markdown
│   ├── chunking.py               # section-aware chunker
│   ├── embeddings.py             # Voyage wrapper + caching
│   ├── indexes.py                # VectorIndex, BM25Index, Retriever
│   ├── paper.py                  # Paper class (the central data object)
│   ├── corpus.py                 # PaperCorpus — manages all papers
│   ├── summarizer.py             # Summarizer class
│   ├── conversation.py           # Conversation class
│   ├── tools.py                  # tool schemas + dispatchers (web search, retrieval)
│   ├── prompts.py                # PromptTemplate + PromptRegistry
│   ├── grader.py                 # Grader class
│   └── pipeline.py               # PromptOptimizationPipeline
├── scripts/
│   ├── ingest_papers.py          # batch parse + chunk + embed
│   ├── summarize.py              # summarize one or all papers
│   ├── ask.py                    # interactive Q&A REPL
│   ├── optimize_prompts.py       # prompt engineering pipeline runner
│   └── grade.py                  # standalone grading on a saved run
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Core Classes

### `Paper` (`paper.py`)

The central data object. Owns its content, summary, and indexes. Lazy throughout.

```python
class Paper:
    paper_id: str                 # content hash of source PDF
    source_path: Path             # original PDF
    title: str | None
    authors: list[str] | None
    year: int | None

    # Lazy-loaded properties (each checks cache, computes if missing, caches result):
    @property
    def text(self) -> str: ...                      # parsed markdown
    @property
    def chunks(self) -> list[Chunk]: ...            # section-aware chunks
    @property
    def summary(self) -> StructuredSummary: ...     # see schema below
    @property
    def chunk_index(self) -> Retriever: ...         # per-paper hybrid retriever
```

`StructuredSummary` is a dataclass with: `tldr` (1-2 sentences), `problem`, `method`, `key_results`, `limitations`, `contributions` (list), `keywords` (list). Generating it as structured output (not free prose) embeds better and is easier to consume downstream.

### `PaperCorpus` (`corpus.py`)

Manages the full collection of papers and exposes corpus-level retrieval.

```python
class PaperCorpus:
    papers: dict[str, Paper]
    summary_index: Retriever      # indexes paper summaries (BM25 + Voyage)

    def add_paper(self, pdf_path: Path) -> Paper: ...
    def get(self, paper_id: str) -> Paper: ...
    def find_relevant_papers(self, query: str, k: int = 5) -> list[Paper]: ...
    def search_chunks(
        self, query: str, k_papers: int = 5, k_chunks: int = 10
    ) -> list[ChunkResult]: ...   # two-tier: filter by summary, then chunks
```

### `Summarizer` (`summarizer.py`)

```python
class Summarizer:
    def __init__(self, client, model="claude-sonnet-4-5", prompt_template=None): ...
    def summarize(self, paper: Paper, force: bool = False) -> StructuredSummary:
        # 1. If paper.summary cache exists and not force → return it
        # 2. Otherwise call LLM with full text (or map-reduce if too long)
        # 3. Parse into StructuredSummary, cache, return
```

For papers that exceed the context window: chunk the paper, summarize each section with a "section summary" prompt, then run a final reduce step that produces the structured summary. Cache section summaries too so re-runs are cheap.

### `Conversation` (`conversation.py`)

Owns a multi-turn chat with tool-use support. Modeled after the user's `using_tools.ipynb` patterns but as a class.

```python
class Conversation:
    def __init__(
        self,
        client,
        corpus: PaperCorpus,
        model: str = "claude-sonnet-4-5",
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        conversation_id: str | None = None,  # resume from SQLite if provided
    ): ...

    messages: list[dict]
    tools: list[Tool]

    def ask(self, user_message: str) -> str:
        # Run the agentic loop:
        # - append user message
        # - call client.messages.create
        # - if stop_reason == "tool_use": dispatch tools, append tool_result, loop
        # - else return text
        # Persist to SQLite after each turn
```

The tool loop must handle: `retrieve_from_papers` (corpus search), `read_paper_summary`, `read_paper_section`, `web_search` (Anthropic's hosted `web_search_20250305` tool). The Conversation class doesn't hardcode tools — it accepts a list, so the prompt-optimization pipeline can swap them.

### `Retriever`, `VectorIndex`, `BM25Index` (`indexes.py`)

Port the user's existing implementations from `chunking_embed.ipynb` directly. Add:
- Pickle save/load for both index types so they can be cached per-paper.
- A `corpus_filter` parameter on `search` so corpus-level chunk search can restrict to a subset of `paper_id`s (used by the two-tier flow).

### `Embedder` (`embeddings.py`)

Thin wrapper around the user's `generate_embedding`. Adds:
- A disk cache keyed by `(content_hash, model, input_type)` so re-running ingestion never re-embeds unchanged chunks.
- Batch handling that respects Voyage rate limits (the user already noted this in their notebook comments).

### `PromptTemplate` and `PromptRegistry` (`prompts.py`)

```python
@dataclass
class PromptTemplate:
    name: str
    version: str          # semantic version, e.g. "qa.v3"
    system: str
    user_template: str    # f-string-like template
    metadata: dict        # author, date, notes
    def render(self, **kwargs) -> tuple[str, str]: ...  # (system, user)

class PromptRegistry:
    def load(self, name: str, version: str = "latest") -> PromptTemplate: ...
    def list_versions(self, name: str) -> list[str]: ...
    def save(self, template: PromptTemplate) -> None: ...
```

Prompts live as `.md` files with YAML frontmatter under `prompts/`. The registry just reads and writes those files. This makes prompts diffable in git.

### `Grader` (`grader.py`)

LLM-as-judge with structured output.

```python
class Grader:
    def __init__(self, client, model="claude-sonnet-4-5", rubric=None): ...

    def grade(
        self,
        question: str,
        answer: str,
        ground_truth: str | None = None,
        retrieved_context: list[str] | None = None,
    ) -> GradeResult:
        # GradeResult fields:
        #   faithfulness: 1-5 (does answer follow from context, no hallucinations)
        #   relevance: 1-5 (does answer address the question)
        #   completeness: 1-5 (covers key points in ground truth)
        #   citation_quality: 1-5 (proper attribution to papers)
        #   overall: weighted average
        #   rationale: str
```

The grader prompt should ask for JSON output and the class parses it (with a retry on parse failure). Use a different model from the one being graded when feasible to reduce self-preference bias.

### `PromptOptimizationPipeline` (`pipeline.py`)

```python
class PromptOptimizationPipeline:
    def __init__(
        self,
        corpus: PaperCorpus,
        eval_dataset: EvalDataset,    # list of {question, ground_truth, relevant_paper_ids}
        grader: Grader,
        candidate_prompts: list[PromptTemplate],
        runner_factory: Callable,     # builds a Conversation given a prompt
    ): ...

    def run(self) -> RunResult:
        # For each prompt × question:
        #   1. Build a fresh Conversation with that prompt
        #   2. Ask the question
        #   3. Grade the response
        # Aggregate per-prompt: mean scores, variance, examples of failures
        # Save full transcripts + scores to eval/runs/{timestamp}/
```

Output a leaderboard markdown comparing prompts by mean overall score, plus the worst examples for each prompt to inspect failure modes.

---

## Two-Tier Retrieval Flow (the answer to last turn's design question)

Implemented in `PaperCorpus.search_chunks`:

1. Embed query.
2. Search `summary_index` → top `k_papers` candidate papers (e.g., 5).
3. For each candidate paper, search its `chunk_index` → top `k_chunks/k_papers` chunks.
4. Merge chunk results, re-rank by RRF across (summary score, chunk score), return top `k_chunks`.
5. Each result carries metadata: `paper_id`, `paper_title`, `section`, `chunk_index`.

The Conversation's `retrieve_from_papers` tool calls this. The result is formatted into the tool_result with paper attribution so Claude can cite cleanly.

---

## Tools (for the Conversation's tool loop)

All tool schemas live in `tools.py` alongside their dispatcher functions, following the `using_tools.ipynb` pattern.

1. **`retrieve_from_papers`** — input: `query`, `k`. Returns top chunks with paper attribution.
2. **`read_paper_summary`** — input: `paper_id`. Returns the structured summary. Cheap; encourages the model to skim before diving.
3. **`read_paper_section`** — input: `paper_id`, `section_name` (or `chunk_index`). Returns full section text. Use when retrieval surfaced a chunk that needs more context.
4. **`list_papers`** — returns paper_id, title, authors, year for everything in the corpus. Lets the model orient itself.
5. **`web_search`** — Anthropic's hosted `web_search_20250305` tool. Configure `max_uses` to bound cost. The user's `using_web_search.ipynb` shows the schema. Allow domain restrictions via config (defaults to none for general research; user can restrict to e.g. `arxiv.org`, `nih.gov`).

The system prompt for the Q&A Conversation should explicitly instruct: prefer `retrieve_from_papers` and `read_paper_summary` first; use `web_search` only when the corpus genuinely lacks coverage; always cite which papers (or web sources) statements come from.

---

## Scripts (entry points)

### `scripts/ingest_papers.py`
- Walks `papers/`, skips already-ingested PDFs (cache check by content hash).
- For each new PDF: parse → chunk → embed chunks → build BM25 → cache.
- Optionally generates summaries in the same pass (`--summarize` flag) — otherwise the summarizer is lazy on first request.
- Updates the corpus-level summary index.

### `scripts/summarize.py`
```
python -m scripts.summarize --paper <paper_id_or_filename>
python -m scripts.summarize --all
python -m scripts.summarize --paper <id> --force   # bypass cache
```
Prints the structured summary; writes JSON to cache. Lazy by default.

### `scripts/ask.py`
Interactive REPL for Q&A.
```
python -m scripts.ask                              # new conversation
python -m scripts.ask --resume <conversation_id>   # continue prior chat
python -m scripts.ask --prompt qa.v3               # use specific prompt version
```
Prints the conversation as it goes, with tool calls visible (collapsed by default, expandable with a flag). Saves to `conversations/`.

### `scripts/optimize_prompts.py`
```
python -m scripts.optimize_prompts \
  --task qa \
  --candidates qa.v1,qa.v2,qa.v3 \
  --eval-set eval/datasets/research_qa.json
```
Runs the `PromptOptimizationPipeline`, writes results to `eval/runs/{timestamp}/`, prints leaderboard.

### `scripts/grade.py`
Standalone grader for an existing run or ad-hoc Q&A pair. Useful for one-off "how did this answer do?" checks without a full pipeline run.

---

## Caching Strategy

Every cached artifact has a content-hash key on its source. Cache lookup:

| Artifact | Key | Format | Location |
|---|---|---|---|
| Parsed text | `sha256(pdf_bytes)` | `.md` | `cache/parsed/` |
| Chunks | `sha256(parsed_text)` | `.json` | embedded in paper meta |
| Chunk embeddings | `sha256(chunk_text) + model` | `.pkl` | `cache/embeddings/` |
| BM25 state | `sha256(parsed_text)` | `.pkl` | `cache/bm25/` |
| Summary | `sha256(parsed_text) + prompt_version` | `.json` | `cache/summaries/` |
| Corpus summary index | hash of all paper_ids | `.pkl` | `cache/corpus_index/` |

`Paper` and `PaperCorpus` consult these caches in their lazy properties. A `--force` flag on every script bypasses cache. A separate `scripts/cache_clear.py` for bulk invalidation.

Note that summary cache keys include the prompt version — this is important so the prompt optimization pipeline doesn't get poisoned by stale summaries from old prompts.

---

## Eval Dataset Format

```json
{
  "name": "research_qa_v1",
  "items": [
    {
      "id": "q001",
      "question": "What architectures have been shown effective for binding site prediction with ESM embeddings?",
      "ground_truth": "Both LSTM-based and 1D CNN architectures (including ConvNeXt-style) have been used; recent work suggests CRF layers help with structured prediction.",
      "relevant_paper_ids": ["paper_abc123", "paper_def456"],
      "category": "synthesis"
    }
  ]
}
```

Categories help break down grader scores by question type (factual lookup vs. synthesis vs. comparative).

---

## Implementation Order (suggested for the coding agent)

1. **Skeleton + config** — repo layout, `config.py`, `.env.example`, `pyproject.toml`.
2. **Ingestion path** — `ingest.py`, `chunking.py`, `embeddings.py` (with caching), `indexes.py` (port from notebook), `Paper` class.
3. **Corpus + two-tier retrieval** — `PaperCorpus`, `scripts/ingest_papers.py`. Verify retrieval works end-to-end before touching the LLM.
4. **Summarizer** — `Summarizer`, `prompts/summarization/v1.md`, `scripts/summarize.py`.
5. **Conversation + tools** — `Conversation`, `tools.py`, `scripts/ask.py`. Wire web search.
6. **Prompts + grader** — `PromptTemplate`, `PromptRegistry`, `Grader`, `prompts/grading/v1.md`.
7. **Pipeline** — `PromptOptimizationPipeline`, `scripts/optimize_prompts.py`, `scripts/grade.py`.
8. **Tests** — unit tests for index ops, caching keys, tool dispatcher, and a smoke test for the conversation loop with mocked clients.

Each phase should leave the system in a working state; don't build phase 7 before phase 3 is verified.

---

## Open Choices to Confirm Before Building

- **Section detection robustness.** The notebook splits on `\n## ` which assumes already-markdown-formatted text. PDF→markdown conversion is imperfect. The agent should add a fallback: if `chunk_by_section` produces only one chunk, switch to fixed-window chunking with overlap (e.g., 1000 tokens, 200 overlap).
- **Re-ranking.** Two-tier already gives a quality boost. A cross-encoder re-ranker (e.g., Voyage's rerank model) could be added as an optional final stage in `Retriever.search`. Mark as a v2 enhancement, scaffold the interface but don't implement initially.
- **Conversation memory beyond messages.** For long research sessions, consider summarizing older turns to keep context manageable. Defer to v2 unless context overflow shows up early.
