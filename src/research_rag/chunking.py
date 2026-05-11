"""TEI-aware chunking for research papers.

Consumes the TEI XML produced by GROBID, walks ``<body>`` divs, and emits a
list of ``Chunk`` objects suitable for embedding/BM25. Long sections are
split into overlapping word-windows; boilerplate sections (references,
acknowledgements, etc.) are dropped.

Public surface:
    Chunk            - dataclass returned for each chunk
    chunk_paper(...) - main entry point
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bs4 import BeautifulSoup


# Default sizing knobs (words). 700 words ≈ 900 tokens for English text.
DEFAULT_TARGET_WORDS = 700
DEFAULT_MAX_WORDS = 1000
DEFAULT_OVERLAP_WORDS = 100


# Section names whose chunks we drop. Matched case-insensitively against
# the head text after stripping trailing punctuation.
_DROP_SECTIONS = frozenset({
    "references",
    "reference",
    "bibliography",
    "works cited",
    "literature cited",
    "acknowledgements",
    "acknowledgments",
    "acknowledgement",
    "acknowledgment",
    "funding",
    "author contributions",
    "conflicts of interest",
    "conflict of interest",
    "competing interests",
    "disclosure",
    "disclosures",
    "data availability",
    "data availability statement",
    "informed consent statement",
    "institutional review board statement",
    "supplementary material",
    "supplementary materials",
    "supporting information",
})


@dataclass
class Chunk:
    content: str
    section: str | None
    subsection: str | None
    chunk_index: int
    char_start: int
    char_end: int
    paper_id: str | None
    metadata: dict = field(default_factory=dict)


def _is_drop(head: str) -> bool:
    return head.strip().lower().rstrip(".:") in _DROP_SECTIONS


def _div_text(div) -> str:
    """Concatenate paragraph text from a div, ignoring nested divs."""
    paras: list[str] = []
    for p in div.find_all("p", recursive=False):
        t = p.get_text(" ", strip=True)
        if t:
            paras.append(t)
    return "\n\n".join(paras)


def _window(text: str, target: int, overlap: int) -> list[str]:
    """Split text into overlapping word-windows."""
    words = text.split()
    if not words:
        return []
    if len(words) <= target:
        return [text]
    if overlap < 0 or overlap >= target:
        raise ValueError("overlap_words must be in [0, target_words)")
    out: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target, len(words))
        out.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return out


def _walk_div(
    div,
    section: str | None,
    subsection: str | None,
    drop_boilerplate: bool,
    out: list[tuple[str, str | None, str | None]],
) -> None:
    """Recursive walk: emit (text, section, subsection) entries."""
    head = div.find("head", recursive=False)
    head_text = head.get_text(" ", strip=True) if head else ""
    if drop_boilerplate and head_text and _is_drop(head_text):
        return

    if section is None:
        new_section = head_text or None
        new_subsection = subsection
    else:
        new_section = section
        new_subsection = head_text or subsection

    text = _div_text(div)
    if text:
        out.append((text, new_section, new_subsection))

    for child in div.find_all("div", recursive=False):
        _walk_div(child, new_section, new_subsection, drop_boilerplate, out)


def chunk_paper(
    tei_xml: str,
    paper_id: str | None = None,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    drop_boilerplate: bool = True,
) -> list[Chunk]:
    """Chunk a GROBID TEI XML document into embedding-ready pieces.

    Strategy:
        1. Emit one chunk for the abstract (section="Abstract").
        2. For each body ``<div>``, take the head as section, paragraphs
           as content. Nested divs become subsections.
        3. Drop divs whose head matches a known boilerplate name.
        4. Split any chunk over ``max_words`` into overlapping word-
           windows of ``target_words`` with ``overlap_words`` overlap.
    """
    if not tei_xml or not tei_xml.strip():
        return []
    if target_words <= 0 or max_words < target_words:
        raise ValueError("invalid sizing parameters")

    soup = BeautifulSoup(tei_xml, "lxml-xml")
    raw: list[tuple[str, str | None, str | None]] = []

    abstract = soup.select_one("teiHeader profileDesc abstract")
    if abstract:
        abs_paras = [p.get_text(" ", strip=True) for p in abstract.find_all("p")]
        abs_paras = [p for p in abs_paras if p]
        if abs_paras:
            raw.append(("\n\n".join(abs_paras), "Abstract", None))

    body = soup.select_one("text body")
    if body:
        for div in body.find_all("div", recursive=False):
            _walk_div(div, None, None, drop_boilerplate, raw)

    chunks: list[Chunk] = []
    idx = 0
    for text, section, subsection in raw:
        word_count = len(text.split())
        pieces = (
            _window(text, target_words, overlap_words)
            if word_count > max_words
            else [text]
        )
        for piece in pieces:
            chunks.append(Chunk(
                content=piece,
                section=section,
                subsection=subsection,
                chunk_index=idx,
                char_start=0,
                char_end=len(piece),
                paper_id=paper_id,
            ))
            idx += 1
    return chunks
