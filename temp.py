"""Interactive Q&A REPL over the paper corpus, with optional web_search."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from anthropic import Anthropic

from src.research_rag import Embedder, PromptTemplate
from src.research_rag.config import (
    DEFAULT_INFERENCE_MODEL,
    PAPERS_DIR,
    PROMPTS_DIR,
    ensure_dirs,
)
from src.research_rag.conversation import Conversation
from src.research_rag.corpus import PaperCorpus
from src.research_rag.tools import make_corpus_tools, web_search_tool




task, version = 'qa', 'v1'
prompt_path = PROMPTS_DIR / task / f"{version}.md"
prompt = PromptTemplate.from_file(prompt_path)

embedder = Embedder()
corpus = PaperCorpus(embedder=embedder, papers_dir=PAPERS_DIR)
corpus.discover()

tools = make_corpus_tools(corpus)

conv = Conversation(
    client=Anthropic(),
    corpus=corpus,
    model=DEFAULT_INFERENCE_MODEL,
    system_prompt=prompt.system,
    tools=tools,
)

handler = conv._tool_handlers.get('retrieve_from_papers')
raw = handler('DCP des-gamma-carboxyprothrombin peptide epitope antibody ELISA HCC hepatocellular carcinoma')

raw = handler('dcp')

