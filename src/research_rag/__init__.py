"""research_rag — local RAG system for scientific papers."""

from .chunking import Chunk, chunk_by_section, chunk_fixed_window, chunk_paper
from .conversation import Conversation
from .corpus import ChunkResult, PaperCorpus
from .embeddings import Embedder
from .grader import GradeResult, Grader, GradingError
from .indexes import BM25Index, Retriever, VectorIndex
from .ingest import PaperIngestionError, hash_pdf, parse_pdf
from .paper import Paper, StructuredSummary
from .pipeline import (
    EvalDataset,
    EvalItem,
    PromptOptimizationPipeline,
    PromptStats,
    RunItemResult,
    RunResult,
)
from .prompts import PromptRegistry, PromptTemplate
from .summarizer import SummarizationError, Summarizer
from .tools import Tool, make_corpus_tools, web_search_tool

__all__ = [
    "BM25Index",
    "Chunk",
    "ChunkResult",
    "Conversation",
    "Embedder",
    "EvalDataset",
    "EvalItem",
    "GradeResult",
    "Grader",
    "GradingError",
    "Paper",
    "PaperCorpus",
    "PaperIngestionError",
    "PromptOptimizationPipeline",
    "PromptRegistry",
    "PromptStats",
    "PromptTemplate",
    "Retriever",
    "RunItemResult",
    "RunResult",
    "StructuredSummary",
    "SummarizationError",
    "Summarizer",
    "Tool",
    "VectorIndex",
    "chunk_by_section",
    "chunk_fixed_window",
    "chunk_paper",
    "hash_pdf",
    "parse_pdf",
    "make_corpus_tools",
    "web_search_tool",
]
