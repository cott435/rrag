# Research RAG System — Planning Document

## Project Overview

A local, domain-pluggable RAG framework. The system currently supports three document domains, each with its own isolated corpus, cache, prompts, and summary schema:

1. **Research** — scientific papers (arXiv, journal articles).
2. **Financial** — earnings reports, 10-K/10-Q filings, investor materials.
3. **Legal** — bills, statutes, regulations, legal articles.

Within a selected domain, the system has two primary user-facing capabilities:

1. **Document Summarization** — generate (and cache) summaries of individual documents on demand, using the domain-specific summary schema.
2. **Domain Q&A** — answer questions using the chosen domain's corpus as the primary knowledge source, supplemented by web search when corpus coverage is insufficient. A session is **pinned to one domain** — there is no cross-domain retrieval in v1.

A secondary capability is a **prompt engineering pipeline** with a grader so the user can iterate on prompts and measure quality across a held-out set of documents and questions, per domain.

### Design Principles

- **Domain-pluggable.** Every domain-specific behavior (parser, chunker config, summary schema, prompts, system-prompt extras, web-search domain whitelist) lives behind a `DomainProfile`. Adding a new domain is a matter of registering a new profile, not editing core classes.
- **Domain isolation.** Each domain has its own document root (`papers/{domain}/`), its own cache namespace (`cache/{domain}/`), and its own prompt namespace (`prompts/{domain}/`). Indexes never mix documents across domains.
- **Lazy by default.** Summaries, embeddings, and BM25 indexes are computed once and cached to disk. On every run, check cache first; only compute what's missing. Cache invalidation keys on a content hash of the source document plus any prompt/config version that influences output.
- **Class-based, composable architecture.** Conversations and retrieval components are objects that own their state. Scripts are thin entry points that wire classes together.
- **Two-tier retrieval.** Both document-level summary embeddings and chunk-level embeddings are indexed per corpus. Retrieval is summary-first, then chunks within selected documents (see Architecture).
- **Hybrid search.** Every retrieval layer runs both dense (Voyage) and BM25, fused with Reciprocal Rank Fusion — same pattern as the user's existing `chunking_embed.ipynb`.
- **Web search is a tool, not a fallback.** Claude decides when to invoke it via tool use, with citations preserved. Each domain configures its own preferred `allowed_domains`.

---

## Domain Model

A `DomainProfile` is the central abstraction that makes the framework multi-domain. It bundles everything domain-specific in one record:

```python
@dataclass
class DomainProfile:
    name: str                          # "research" | "financial" | "legal"
    documents_dir: Path                # papers/{name}/
    cache_root: Path                   # cache/{name}/
    parser: DocumentParser             # protocol; GROBID for research, TBD for others
    chunker_config: ChunkerConfig      # drop-sections list, target window, overlap, etc.
    summary_schema: type[BaseSummary]  # the dataclass for this domain
    prompt_namespace: str              # prompts/{name}/
    system_prompt_extras: str          # appended to the Q&A system prompt
    web_search_domains: list[str]     # default allowed_domains for web_search tool
```

Profiles live in `src/research_rag/domains/` — one file per domain. A central `DomainRegistry` loads profiles by name. All user-facing scripts accept `--domain` and resolve through the registry.

As part of this generalization, the core data types are renamed:

- `Paper` → `Document` (constructor takes a `DomainProfile`)
- `PaperCorpus` → `Corpus` (bound to a single `DomainProfile`)
- `StructuredSummary` → `BaseSummary` with one subclass per domain (`ResearchSummary`, `FinancialSummary`, `LegalSummary`)

No back-compat aliases — the codebase is small enough to rename cleanly.

---

## Tech Stack

- **Language:** Python 3.10+
- **LLM:** Anthropic Claude via the `anthropic` SDK. Default to `claude-sonnet-4-5` for inference and grading; `claude-haiku-4-5` for cheaper bulk operations like summarization of many documents (configurable per-class).
- **Embeddings:** VoyageAI (`voyage-3-large`), shared across all domains. Domain-tuned embeddings (FinBERT, LegalBERT) are flagged as a v2 option behind the same embedder interface.
- **Local LLM fallback:** Ollama for offline summarization (Qwen).

### Parsing

- **Research domain:** GROBID (`/api/processFulltextDocument`), producing TEI XML. Already integrated.
- **Financial domain:** TBD — see Open Choices. Candidates include XBRL/iXBRL parsers (`Arelle`) for SEC filings and layout-aware PDF tools (Unstructured.io, pdfplumber + heuristics) for earnings PDFs.
- **Legal domain:** TBD — see Open Choices. Candidates include direct USLM XML parsing for US bills/statutes, and `eyecite` for citation extraction.
- **Generic fallback:** `pypdf` / `pdfplumber` for any domain that doesn't yet have a specialized parser. `pytesseract` only behind an explicit `--ocr` flag.

The embedding/index/retrieval stack is domain-agnostic — the same Voyage + BM25 + RRF pipeline runs for every domain.

- **Storage:** Local filesystem only. JSON for metadata and summaries, pickle for embedding indexes, SQLite for conversation history. No external DB.
- **Env:** `python-dotenv` for `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY`.
- **Testing:** `pytest`. Mock LLM/embedding calls.

---

## Repository Layout

```
research_rag/
├── papers/
│   ├── research/                # user drops research PDFs here
│   ├── financial/               # user drops filings, earnings PDFs here
│   └── legal/                   # user drops bills, statutes here
├── cache/
│   ├── research/
│   │   ├── parsed/              # {doc_id}.tei.xml + {doc_id}.md
│   │   ├── chunks/              # {doc_id}.{chunker_config_hash}.json
│   │   ├── summaries/           # {doc_id}.{prompt_version}.json
│   │   ├── bm25/                # {doc_id}.pkl
│   │   └── corpus_index/        # corpus-level summary index
│   ├── financial/               # (same subtree)
│   ├── legal/                   # (same subtree)
│   └── embeddings/              # SHARED across domains — keyed by content hash
├── conversations/               # SQLite DB; each row tagged with domain
├── prompts/
│   ├── research/
│   │   ├── summarization/
│   │   ├── summarization_chain/
│   │   ├── qa/
│   │   └── grading/
│   ├── financial/               # (same subtree)
│   └── legal/                   # (same subtree)
├── eval/
│   ├── datasets/                # {domain}_qa_v1.json
│   └── runs/                    # timestamped eval results
├── src/research_rag/
│   ├── __init__.py
│   ├── config.py                # paths, model defaults
│   ├── domains/
│   │   ├── __init__.py          # DomainRegistry
│   │   ├── base.py              # DomainProfile, DocumentParser, BaseSummary, ChunkerConfig
│   │   ├── research.py          # research profile, GROBID parser, ResearchSummary
│   │   ├── financial.py         # financial profile, parser TBD, FinancialSummary
│   │   └── legal.py             # legal profile, parser TBD, LegalSummary
│   ├── ingest.py                # generic parser dispatch (delegates to domain parser)
│   ├── chunking.py              # generic chunker over Section IR + ChunkerConfig
│   ├── embeddings.py            # Voyage wrapper + caching (shared)
│   ├── indexes.py               # VectorIndex, BM25Index, Retriever (shared)
│   ├── document.py              # Document class (was paper.py)
│   ├── corpus.py                # Corpus, parametrized by DomainProfile
│   ├── summarizer.py            # generic Summarizer; reads schema + per-field guidance from prompt
│   ├── conversation.py          # binds to one Corpus / one Domain
│   ├── tools.py                 # tool schemas + dispatchers (corpus-scoped, domain-aware descriptions)
│   ├── prompts.py               # PromptTemplate + PromptRegistry (namespaced by domain)
│   ├── grader.py                # Grader class
│   └── pipeline.py              # PromptOptimizationPipeline
├── scripts/
│   ├── ingest.py                # --domain {research,financial,legal,all}
│   ├── summarize.py             # --domain required
│   ├── ask.py                   # --domain required; pinned per session
│   ├── optimize_prompts.py      # --domain required
│   └── grade.py                 # standalone grading on a saved run
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

The `cache/embeddings/` directory is intentionally **outside** the per-domain namespaces: chunk embeddings are keyed by `(sha256(chunk_text), model)`, so a chunk that appears in two corpora (rare but possible) is embedded once.

---

## Core Classes

### `Document` (`document.py`)

The central data object (renamed from `Paper`). Owns its content, summary, and indexes. Lazy throughout. Bound to a `DomainProfile` at construction.

```python
class Document:
    document_id: str              # content hash of source file
    source_path: Path             # original file
    profile: DomainProfile        # the domain this document belongs to
    title: str | None
    metadata: dict                # domain-specific fields (authors/year for research, ticker/period for financial, bill_number/jurisdiction for legal)

    # Lazy-loaded properties (each checks cache, computes if missing, caches result):
    @property
    def canonical(self) -> str: ...                 # parser-native form (TEI / XBRL / USLM / ...)
    @property
    def text(self) -> str: ...                      # markdown rendering for display
    @property
    def sections(self) -> list[Section]: ...        # parser-emitted Section IR (see Parser Protocol)
    @property
    def chunks(self) -> list[Chunk]: ...            # chunker output (uses profile.chunker_config)
    @property
    def summary(self) -> BaseSummary: ...           # typed as profile.summary_schema
    @property
    def chunk_index(self) -> Retriever: ...         # per-document hybrid retriever
```

### `BaseSummary` and domain subclasses (`domains/base.py`, `domains/{name}.py`)

```python
@dataclass
class BaseSummary:
    tldr: str
    keywords: list[str]

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "BaseSummary": ...
    def to_embed_text(self) -> str: ...   # how this summary is rendered for the summary index
```

Each domain registers its own subclass with whatever fields make sense:

- **`ResearchSummary`**: `problem`, `method`, `key_results`, `limitations`, `contributions: list[str]`.
- **`FinancialSummary`**: `period`, `key_metrics`, `guidance`, `risks`, `segment_highlights`.
- **`LegalSummary`**: `jurisdiction`, `subject`, `key_provisions`, `affected_statutes`, `status`.

These field sets are first drafts — refine them when implementing each domain. The grader and prompt pipeline read field names from the schema, so adding/removing fields is a localized change.

Generating each summary as structured output (not free prose) keeps embeddings tight and makes downstream consumption easy.

### `Corpus` (`corpus.py`)

Manages the full collection of documents for **one** domain and exposes corpus-level retrieval.

```python
class Corpus:
    profile: DomainProfile
    documents: dict[str, Document]
    summary_index: Retriever              # indexes document summaries (BM25 + Voyage)

    def add_document(self, source_path: Path) -> Document: ...
    def get(self, document_id: str) -> Document: ...
    def find_relevant_documents(self, query: str, k: int = 5) -> list[Document]: ...
    def search_chunks(
        self, query: str, k_docs: int = 5, k_chunks: int = 10
    ) -> list[ChunkResult]: ...   # two-tier: filter by summary, then chunks
```

`_summary_text` (the function that renders a summary for the summary index) moves onto each `BaseSummary` subclass as `to_embed_text()`, so each domain controls how its summary appears to the retriever.

### `Summarizer` (`summarizer.py`)

Now generic. Reads the target schema from the bound domain profile, and reads per-field guidance from the domain's prompt file (rather than hardcoding it in Python).

```python
class Summarizer:
    def __init__(self, client, profile: DomainProfile, model="claude-sonnet-4-5"): ...
    def summarize(self, document: Document, force: bool = False) -> BaseSummary:
        # 1. If document.summary cache exists for this prompt version and not force → return it
        # 2. Otherwise call LLM with full text (or chained per-field reduce if too long)
        # 3. Parse into profile.summary_schema, cache, return
```

For documents that exceed the context window: chunk the document, summarize each section, then run a final reduce step that produces the structured summary. Cache section summaries too so re-runs are cheap. The chained-Ollama path also reads its field list from the schema rather than a hardcoded constant.

### `Conversation` (`conversation.py`)

Owns a multi-turn chat with tool-use support. **Pinned to one domain** — constructed with a `Corpus` that is bound to a `DomainProfile`. The system prompt is composed from a shared base + `profile.system_prompt_extras`.

```python
class Conversation:
    def __init__(
        self,
        client,
        corpus: Corpus,                   # already domain-bound
        model: str = "claude-sonnet-4-5",
        system_prompt: str | None = None, # if None, compose from shared + profile.system_prompt_extras
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
        # Persist to SQLite (including the active domain) after each turn
```

The tool loop must handle: `retrieve_from_corpus`, `read_document_summary`, `read_document_section`, `list_documents`, `web_search` (Anthropic's hosted `web_search_20250305` tool, with `allowed_domains` from the profile). The Conversation class doesn't hardcode tools — it accepts a list, so the prompt-optimization pipeline can swap them.

### `Retriever`, `VectorIndex`, `BM25Index` (`indexes.py`)

Unchanged from the original design. Port the user's existing implementations from `chunking_embed.ipynb` directly. Add:
- Pickle save/load for both index types so they can be cached per-document.
- A `corpus_filter` parameter on `search` so corpus-level chunk search can restrict to a subset of `document_id`s (used by the two-tier flow).

### `Embedder` (`embeddings.py`)

Unchanged from the original design — shared across all domains. The disk cache is keyed by `(content_hash, model, input_type)`, so re-running ingestion never re-embeds unchanged chunks regardless of which domain they belong to. Batch handling respects Voyage rate limits.

### `PromptTemplate` and `PromptRegistry` (`prompts.py`)

The registry is **namespaced by domain**.

```python
@dataclass
class PromptTemplate:
    name: str
    version: str          # semantic version, e.g. "qa.v3"
    domain: str           # "research" | "financial" | "legal"
    system: str
    user_template: str    # f-string-like template
    metadata: dict        # author, date, notes
    def render(self, **kwargs) -> tuple[str, str]: ...  # (system, user)

class PromptRegistry:
    def load(self, domain: str, name: str, version: str = "latest") -> PromptTemplate: ...
    def list_versions(self, domain: str, name: str) -> list[str]: ...
    def save(self, template: PromptTemplate) -> None: ...
```

Prompts live as `.md` files with YAML frontmatter under `prompts/{domain}/{name}/v{n}.md`. The registry just reads and writes those files — no separate index file. Prompts are diffable in git.

### `Grader` (`grader.py`)

LLM-as-judge with structured output. Domain-agnostic in v1 (shared rubric); per-domain rubrics are a v2 option.

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
        #   citation_quality: 1-5 (proper attribution to documents)
        #   overall: weighted average
        #   rationale: str
```

The grader prompt should ask for JSON output and the class parses it (with a retry on parse failure). Use a different model from the one being graded when feasible to reduce self-preference bias.

### `PromptOptimizationPipeline` (`pipeline.py`)

Operates on a single domain at a time (no cross-domain optimization in v1).

```python
class PromptOptimizationPipeline:
    def __init__(
        self,
        corpus: Corpus,                   # domain-bound
        eval_dataset: EvalDataset,        # list of {question, ground_truth, relevant_document_ids}
        grader: Grader,
        candidate_prompts: list[PromptTemplate],
        runner_factory: Callable,         # builds a Conversation given a prompt
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

## Parser Protocol

The parser is the **only** part of ingestion that is fundamentally domain-specific. Every parser implements the same protocol, and the chunker operates on the parser's `Section` output — not on raw XML.

```python
class DocumentParser(Protocol):
    """Turns a source file into (canonical, markdown, sections)."""
    def parse(self, source: Path) -> ParseResult: ...

@dataclass
class ParseResult:
    canonical: str            # parser-native rich form (TEI for GROBID, XBRL for filings, USLM for bills)
    markdown: str             # human-readable rendering, used for display + fallback
    sections: list[Section]   # parser-agnostic IR for the chunker

@dataclass
class Section:
    title: str | None
    kind: str                 # "abstract" | "body" | "schedule" | "amendment" | ...  domain-defined
    level: int                # heading depth
    text: str
```

The chunker is now generic: it walks `list[Section]`, applies `profile.chunker_config` (drop-sections list, target window, overlap), and emits `Chunk` objects. The current TEI-walking logic in `chunking.py` moves into `domains/research.py` as part of the GROBID parser. Drop-section lists are no longer global constants — they live on `ChunkerConfig` per domain.

### Open parser questions

- **Financial:** SEC filings come as XBRL/iXBRL (structured) or as PDFs of earnings reports (unstructured). Candidates to evaluate when implementing: `sec-edgar-downloader` for retrieval; `Arelle` or hand-rolled XBRL parsing for filings; layout-aware PDF (Unstructured.io, pdfplumber + heuristics) for earnings PDFs.
- **Legal:** US bills are published as USLM XML on Congress.gov. Candidates: direct XML parsing for bills/statutes; `eyecite` for citation extraction. PDFs of legal articles may use plain pypdf/pdfplumber. State-level legal documents have no unified standard — generic PDF parser first, specialize as needed.

**GROBID will not work well on financial filings or legal bills.** Its TEI schema and section heuristics are trained on academic publishing layout. Financial and legal domains require different parsers — picked at the start of their respective implementation phases.

---

## Two-Tier Retrieval Flow

Implemented in `Corpus.search_chunks`. Retrieval **never crosses domain boundaries** — each Corpus owns its summary and chunk indexes, and a Conversation queries only its bound Corpus.

1. Embed query.
2. Search the bound corpus's `summary_index` → top `k_docs` candidate documents (e.g., 5).
3. For each candidate document, search its `chunk_index` → top `k_chunks/k_docs` chunks.
4. Merge chunk results, re-rank by RRF across (summary score, chunk score), return top `k_chunks`.
5. Each result carries metadata: `document_id`, `document_title`, `section`, `chunk_index`.

The Conversation's `retrieve_from_corpus` tool calls this. The result is formatted into the tool_result with document attribution so Claude can cite cleanly.

---

## Tools (for the Conversation's tool loop)

All tool schemas live in `tools.py` alongside their dispatcher functions, following the `using_tools.ipynb` pattern. Tools are corpus-scoped — they operate on the Conversation's bound Corpus only.

1. **`retrieve_from_corpus`** — input: `query`, `k`. Returns top chunks with document attribution. Tool description includes the active domain name so Claude has correct context (e.g., "Search the financial corpus...").
2. **`read_document_summary`** — input: `document_id`. Returns the structured summary (typed to the domain's schema). Cheap; encourages the model to skim before diving.
3. **`read_document_section`** — input: `document_id`, `section_name` (or `chunk_index`). Returns full section text. Use when retrieval surfaced a chunk that needs more context.
4. **`list_documents`** — returns `document_id`, `title`, plus domain-specific metadata (authors/year for research, ticker/period for financial, bill_number/jurisdiction for legal) for everything in the bound corpus.
5. **`web_search`** — Anthropic's hosted `web_search_20250305` tool. `max_uses` bounds cost. `allowed_domains` defaults from `profile.web_search_domains` (e.g., `arxiv.org, nih.gov` for research; `sec.gov, *.investor.*` for financial; `congress.gov, govinfo.gov` for legal). Users can override per-session.

The system prompt for the Q&A Conversation is composed from a shared base + `profile.system_prompt_extras`. The shared part instructs: prefer corpus tools first; use `web_search` only when the corpus genuinely lacks coverage; always cite which documents (or web sources) statements come from. The profile extras add domain-specific framing (e.g., "When answering legal questions, always identify the jurisdiction.").

---

## Scripts (entry points)

Every user-facing script accepts a required `--domain {research,financial,legal}` flag (except `grade.py`, which operates on a saved run that already records its domain). `scripts/ingest.py` additionally accepts `--domain all` to walk every registered domain in sequence.

### `scripts/ingest.py`
- Walks `papers/{domain}/`, skips already-ingested files (cache check by content hash).
- For each new file: parse (via `profile.parser`) → chunk (via `profile.chunker_config`) → embed chunks → build BM25 → cache.
- Optionally generates summaries in the same pass (`--summarize` flag) — otherwise the summarizer is lazy on first request.
- Updates the corpus-level summary index for that domain.

### `scripts/summarize.py`
```
python -m scripts.summarize --domain research --document <id_or_filename>
python -m scripts.summarize --domain financial --all
python -m scripts.summarize --domain legal --document <id> --force   # bypass cache
```
Prints the structured summary; writes JSON to `cache/{domain}/summaries/`. Lazy by default.

### `scripts/ask.py`
Interactive REPL for Q&A, pinned to one domain for the session.
```
python -m scripts.ask --domain research                            # new conversation
python -m scripts.ask --domain financial --resume <conv_id>        # continue prior chat (must match domain)
python -m scripts.ask --domain legal --prompt qa.v3                # use specific prompt version
```
REPL banner displays the active domain. Tool calls visible (collapsed by default, expandable with a flag). Saves to `conversations/` SQLite with the domain recorded per row. Resuming a conversation requires the matching `--domain`.

### `scripts/optimize_prompts.py`
```
python -m scripts.optimize_prompts \
  --domain research \
  --task qa \
  --candidates qa.v1,qa.v2,qa.v3 \
  --eval-set eval/datasets/research_qa.json
```
Runs the `PromptOptimizationPipeline` against the named domain, writes results to `eval/runs/{timestamp}/`, prints leaderboard.

### `scripts/grade.py`
Standalone grader for an existing run or ad-hoc Q&A pair. The run's stored domain is used to scope the rubric (v2: per-domain rubrics).

---

## Caching Strategy

Every cached artifact has a content-hash key on its source. Cache lookup:

| Artifact | Key | Format | Location |
|---|---|---|---|
| Parsed canonical | `sha256(source_bytes)` | parser-native (`.tei.xml`, `.xbrl`, `.uslm.xml`, ...) | `cache/{domain}/parsed/` |
| Parsed markdown | `sha256(canonical)` | `.md` | `cache/{domain}/parsed/` |
| Chunks | `sha256(canonical) + chunker_config_hash` | `.json` | `cache/{domain}/chunks/` |
| Chunk embeddings | `sha256(chunk_text) + model` | `.pkl` | `cache/embeddings/` (shared across domains) |
| BM25 state | `sha256(canonical)` | `.pkl` | `cache/{domain}/bm25/` |
| Summary | `sha256(canonical) + domain + prompt_version` | `.json` | `cache/{domain}/summaries/` |
| Corpus summary index | hash of document_ids in corpus | `.pkl` | `cache/{domain}/corpus_index/` |

The `Document` and `Corpus` classes consult these caches in their lazy properties. A `--force` flag on every script bypasses cache. A separate `scripts/cache_clear.py --domain ...` for bulk invalidation, scoped to one domain.

Key invariants worth highlighting:

- **Summary cache keys include the prompt version** — important so the prompt optimization pipeline doesn't get poisoned by stale summaries from old prompts.
- **Chunk cache keys include the chunker config hash** — changing `ChunkerConfig` (e.g., adjusting drop-sections for filings) must invalidate cached chunks. Without this, chunker tuning silently no-ops.
- **Embeddings cache is shared across domains** — chunk embeddings are keyed by content hash + model only, so identical text in different corpora is embedded once.

---

## Eval Dataset Format

```json
{
  "name": "research_qa_v1",
  "domain": "research",
  "items": [
    {
      "id": "q001",
      "question": "What architectures have been shown effective for binding site prediction with ESM embeddings?",
      "ground_truth": "Both LSTM-based and 1D CNN architectures (including ConvNeXt-style) have been used; recent work suggests CRF layers help with structured prediction.",
      "relevant_document_ids": ["doc_abc123", "doc_def456"],
      "category": "synthesis"
    }
  ]
}
```

Filenames follow the convention `{domain}_qa_v1.json` (e.g., `research_qa_v1.json`, `financial_qa_v1.json`, `legal_qa_v1.json`). Categories help break down grader scores by question type (factual lookup vs. synthesis vs. comparative). Per-domain grader rubrics are a v2 concern; v1 uses the shared rubric.

---

## Implementation Order (suggested for the coding agent)

The phases are ordered so the research domain stays fully working at every step. No half-renamed code in master.

1. **Refactor scaffolding** — introduce `DomainProfile`, `DomainRegistry`, `BaseSummary`, `Section` IR, `ChunkerConfig`. Rename `Paper` → `Document`, `PaperCorpus` → `Corpus`, `StructuredSummary` → `ResearchSummary`. Move all current research-paper-specific logic (GROBID parser, TEI walking, research drop-sections, research field instructions) behind `domains/research.py`. **System must still pass tests with the research domain only.**
2. **Cache + path migration** — namespace `cache/{domain}/...` and `papers/{domain}/...`. Provide a one-shot migration script that moves existing files under `cache/research/` and `papers/research/`. Embeddings cache stays at `cache/embeddings/` (shared).
3. **PromptRegistry namespacing** — move existing prompts under `prompts/research/`; update the registry to take a `domain` parameter; update scripts to pass it.
4. **Financial domain** — pick parser (decision point: XBRL vs. layout-aware PDF, see Open Choices); implement `domains/financial.py`; write `FinancialSummary` schema and prompts (summarization, qa, grading); add test fixtures (1–2 small filings or earnings PDFs).
5. **Legal domain** — same shape; pick parser (USLM XML vs. plain PDF); implement `domains/legal.py`; `LegalSummary` schema + prompts; fixtures (1–2 small bills/statutes).
6. **Cross-domain UX polish** — `--domain` flags wired everywhere, REPL banner shows active domain, conversation history table gains a `domain` column with a migration, eval datasets renamed.
7. **Tests** — domain-agnostic unit tests for index ops, caching keys, tool dispatcher; per-domain smoke tests for parser → chunker → summarizer with mocked LLM.

Each phase should leave the system in a working state; don't build phase 6 before phase 4 is verified.

---

## Open Choices to Confirm Before Building

These are flagged in the plan and should not be silently resolved without checking with the user:

- **Financial parser choice** — XBRL/iXBRL via Arelle vs. layout-aware PDF (Unstructured.io / pdfplumber heuristics) vs. both. Decide at the start of Phase 4.
- **Legal parser choice** — USLM XML for federal bills vs. plain PDF for articles and state documents. Decide at the start of Phase 5. May end up with two parsers under `domains/legal.py` and a heuristic to route between them.
- **Section detection robustness for fallback parsers** — for the generic PDF path used until a domain has a specialized parser: if section detection produces only one section, switch to fixed-window chunking with overlap (e.g., 1000 tokens, 200 overlap). The `ChunkerConfig` already supports this.
- **Per-domain embedders** — `DomainProfile` could specify a domain-tuned embedding (FinBERT, LegalBERT). Interface allows it; v1 uses Voyage everywhere.
- **Per-domain grader rubrics** — financial answers care about period attribution; legal answers care about jurisdiction citation. Defer to v2.
- **Cross-domain Q&A** — explicitly out of scope for v1. Each session is pinned to one domain.
- **Re-ranking.** Two-tier already gives a quality boost. A cross-encoder re-ranker (e.g., Voyage's rerank model) could be added as an optional final stage in `Retriever.search`. Mark as a v2 enhancement, scaffold the interface but don't implement initially.
- **Conversation memory beyond messages.** For long research sessions, consider summarizing older turns to keep context manageable. Defer to v2 unless context overflow shows up early.
