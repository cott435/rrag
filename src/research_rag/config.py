"""Project configuration: paths, model defaults, env loading."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths — anchored at the repo root (parent of src/)
ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "papers"
CACHE_DIR = ROOT / "cache"
PARSED_CACHE = CACHE_DIR / "parsed"
SUMMARIES_CACHE = CACHE_DIR / "summaries"
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings"
BM25_CACHE = CACHE_DIR / "bm25"
CORPUS_INDEX_CACHE = CACHE_DIR / "corpus_index"
CONVERSATIONS_DIR = ROOT / "conversations"
PROMPTS_DIR = ROOT / "prompts"
EVAL_DIR = ROOT / "eval"
EVAL_DATASETS = EVAL_DIR / "datasets"
EVAL_RUNS = EVAL_DIR / "runs"

# Model defaults
DEFAULT_INFERENCE_MODEL = "claude-sonnet-4-5"
DEFAULT_BULK_MODEL = "claude-haiku-4-5"
DEFAULT_GRADING_MODEL = "claude-sonnet-4-5"
DEFAULT_EMBEDDING_MODEL = "voyage-4-large"
DEFAULT_LOCAL_MODEL = "qwen3:14b"

# API keys (read after load_dotenv)
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY: str | None = os.getenv("VOYAGE_API_KEY")
OLLAMA_HOST: str | None = os.getenv("OLLAMA_HOST")


def ensure_dirs() -> None:
    """Create cache and output directories that don't exist yet."""
    for d in (
        PAPERS_DIR,
        CACHE_DIR,
        PARSED_CACHE,
        SUMMARIES_CACHE,
        EMBEDDINGS_CACHE,
        BM25_CACHE,
        CORPUS_INDEX_CACHE,
        CONVERSATIONS_DIR,
        EVAL_DATASETS,
        EVAL_RUNS,
    ):
        d.mkdir(parents=True, exist_ok=True)
