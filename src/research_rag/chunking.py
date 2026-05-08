"""Section-aware chunking for research papers, with size-bounded windowing.

Designed for scientific papers extracted from PDF or downloaded as plaintext/
markdown. Tries several heading conventions before falling back to a sliding
window. Output is always a list[Chunk] with reliable section/subsection
metadata and accurate character offsets into the *original* text.

Public surface:
    Chunk            - dataclass returned for each chunk
    chunk_paper(...) - main entry point (called by downstream code)

The lower-level helpers (chunk_by_section, chunk_fixed_window, etc.) are
exposed for tests and for callers that want to compose their own pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target chunk size in words. Roughly 1.3x in tokens for English text, so
# 700 words ≈ 900 tokens, which fits comfortably in 1k-token embedding models.
DEFAULT_TARGET_WORDS = 700
DEFAULT_MAX_WORDS = 1000
DEFAULT_MIN_WORDS = 50
DEFAULT_OVERLAP_WORDS = 100

# Sections we drop entirely. Matched case-insensitively against the heading
# text after stripping numbering. References/bibliography are dropped because
# they're citation noise that hurts retrieval; acknowledgements/funding/
# conflicts are dropped because they're boilerplate.
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


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

# Markdown-style: "## Heading" or "### Heading" (1–4 #s).
_MD_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")

# Numbered: "1 Introduction", "1. Introduction", "3.1 Foo", "3.1.2 Bar baz".
# Allow up to 3 levels of numbering. Heading text must start with a letter to
# avoid matching things like "1. 2020 was a great year".
_NUMBERED_HEADING = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Za-z][^\n]{1,120})$"
)

# ALL CAPS heading: "INTRODUCTION", "MATERIALS AND METHODS". Require at least
# 3 chars, allow spaces/&/-. Avoid matching shouty sentences by capping length
# and disallowing terminal punctuation.
_ALLCAPS_HEADING = re.compile(r"^([A-Z][A-Z0-9 &/\-]{2,80}[A-Z0-9])$")

# Common section names that may appear without numbering or special casing.
# Matched case-insensitively as a whole-line heading.
_NAMED_HEADINGS = frozenset({
    "abstract",
    "introduction",
    "background",
    "related work",
    "related works",
    "methods",
    "method",
    "methodology",
    "materials and methods",
    "materials & methods",
    "experimental setup",
    "experiments",
    "empirical experiments",
    "results",
    "results and discussion",
    "discussion",
    "conclusion",
    "conclusions",
    "conclusions and future work",
    "summary",
    "limitations",
    "future work",
    "review",
    "review method",
    "appendix",
    # Terminal/boilerplate headings — included here so they get detected as
    # headings (a prerequisite for dropping them downstream).
    "references",
    "reference",
    "bibliography",
    "works cited",
    "literature cited",
    "acknowledgements",
    "acknowledgments",
    "funding",
    "author contributions",
    "conflicts of interest",
    "conflict of interest",
    "competing interests",
    "disclosure",
    "disclosures",
    "data availability",
})


def _classify_line(line: str) -> tuple[int, str, str | None] | None:
    """Try to parse a line as a heading.

    Returns (level, title, numbering) on match, where:
        level     - 1 (top-level section), 2 (subsection), or 3 (subsubsection)
        title     - heading text without numbering
        numbering - the numbering prefix (e.g., "3.1") or None

    Returns None if the line is not a heading.
    """
    s = line.strip()
    if not s:
        return None

    # Markdown headings.
    m = _MD_HEADING.match(s)
    if m:
        hashes, title = m.group(1), m.group(2).strip()
        # `#` and `##` -> level 1; `###` -> level 2; `####` -> level 3.
        level = 1 if len(hashes) <= 2 else (2 if len(hashes) == 3 else 3)
        return (level, title, None)

    # Numbered headings.
    m = _NUMBERED_HEADING.match(s)
    if m:
        numbering = m.group(1)
        title = m.group(2).strip()
        # Trailing periods and trailing-asterisk footnote markers are common
        # in extracted PDFs ("Introduction*"). Strip them.
        title = title.rstrip(" .*")
        # Citations in numbered reference lists ("1 Smith J, Doe A. Title.")
        # have the same syntactic shape as numbered headings. Reject the
        # match if the title contains a citation-style "Surname X," pattern
        # — a capitalized word followed by 1-3 capital initials and a comma.
        if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z]{1,3})+,", title):
            return None
        # Section titles also rarely contain a semicolon followed by a year.
        if re.search(r";\s*\d{4}", title):
            return None
        # Reject titles that are just a date or a year — these come from
        # journal mastheads ("1 July 2017") that the PDF extractor leaves
        # as standalone lines.
        _MONTHS = (
            "January|February|March|April|May|June|July|August|September|"
            "October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
        )
        if re.fullmatch(rf"(?:{_MONTHS})\s+\d{{4}}", title, re.IGNORECASE):
            return None
        if re.fullmatch(r"\d+", title):
            return None
        # Reject titles that look like running text or formula fragments
        # rather than headings. Numbered citations and column-flow artifacts
        # ("[42] Therefore, ...") routinely produce these false positives.
        if len(title) > 100 or len(title.split()) > 12:
            return None
        if any(ch in title for ch in ",()/;"):
            return None
        # Reject titles dominated by single-letter tokens (PDF table-row
        # extraction artifacts like "0 E E E EE E E EEE").
        toks = title.split()
        if toks and sum(1 for t in toks if len(t) <= 1) >= max(2, len(toks) // 2):
            return None
        # Real numbered section titles begin with a capitalized content word
        # ("3.1 Architecture", "8 Conclusion And Future Work"). Sentence-case
        # fragments like "13 In our previous" pass that test but are still
        # almost always citation continuations or running text — but those
        # have already been filtered above by the comma/length rules. The
        # remaining "13 In our previous"-style cases happen when extracted
        # PDF text has no comma; for those we additionally require that the
        # title isn't a sentence fragment by checking the SECOND content word
        # too. Stopwords ("in", "of", "the", "and", ...) don't count.
        _TITLE_STOPWORDS = {
            "a", "an", "and", "as", "at", "but", "by", "for", "from", "in",
            "into", "of", "on", "or", "the", "to", "via", "with", "vs",
        }
        content_words = [t for t in toks if t.lower() not in _TITLE_STOPWORDS]
        if not content_words:
            return None
        if not content_words[0][0:1].isupper():
            return None
        # If there are multiple content words, at least half must start with
        # a capital letter — kills "13 In our previous" (1/2 capitalized
        # content words, fails 0.5 threshold? "previous" is content → 1/2
        # = 50%; we need strictly more than 50%, so use >= 0.6).
        if len(content_words) >= 2:
            cap_ratio = sum(1 for t in content_words if t[0:1].isupper()) / len(content_words)
            if cap_ratio < 0.6:
                return None
        # Normalize all-caps titles to title case for consistency.
        if title.isupper() and len(title) > 3:
            title = title.title()
        depth = numbering.count(".") + 1
        level = min(depth, 3)
        return (level, title, numbering)

    # Named headings (case-insensitive whole-line match).
    canonical = s.lower().rstrip(".:")
    if canonical in _NAMED_HEADINGS:
        # Normalize casing: if the source was all-caps or all-lower, present
        # it title-cased; otherwise preserve whatever casing was used.
        cleaned = s.rstrip(".:")
        if cleaned.isupper() or cleaned.islower():
            cleaned = cleaned.title()
        return (1, cleaned, None)

    # ALL CAPS as a last resort. Only trust short standalone lines.
    if len(s) <= 80 and _ALLCAPS_HEADING.match(s):
        # Avoid matching ALL-CAPS sentences inside running text by requiring
        # the line to be reasonably short word-wise.
        words = s.split()
        if len(words) <= 8:
            # Reject if no internal token has at least 4 alphabetic characters
            # — that filters out table column headers like "EN-DE EN-FR" and
            # standalone abbreviations like "DO X". Split on whitespace AND
            # hyphens so "EN-DE" doesn't masquerade as a 4-letter word.
            sub_tokens = re.split(r"[\s\-]+", s)
            longest_alpha = max(
                (sum(1 for ch in t if ch.isalpha()) for t in sub_tokens),
                default=0,
            )
            if longest_alpha < 4:
                return None
            # Normalize to title case so downstream consumers see consistent
            # casing regardless of how the heading appeared in the source.
            return (1, s.title(), None)

    return None


def _is_drop_heading(title: str) -> bool:
    """Should chunks under this heading be discarded?"""
    return title.strip().lower().rstrip(".:") in _DROP_SECTIONS


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    """A contiguous span of text under one section/subsection.

    Internal representation; converted to one or more `Chunk`s downstream.
    """
    section: str | None
    subsection: str | None
    char_start: int
    char_end: int
    text: str


def _parse_blocks(text: str) -> list[_Block]:
    """Walk the text line-by-line, grouping body lines into blocks keyed by
    the most recent (section, subsection) heading.

    Char offsets are tracked against the original `text` so downstream chunks
    can be located exactly.
    """
    if not text:
        return []

    blocks: list[_Block] = []
    cur_section: str | None = None
    cur_subsection: str | None = None
    buf: list[str] = []
    buf_start: int | None = None  # char offset of start of current buffer
    buf_end: int = 0  # char offset just past the last buffered char
    # Once we enter References / Bibliography, treat the rest of the document
    # as body for the current section. Numbered citation entries
    # ("1 Smith J, ...") otherwise mimic numbered headings and produce dozens
    # of spurious sections. If a real heading appears later (e.g., Appendix),
    # we still want to pick it up — so we only suppress *numbered* headings
    # while in this mode, allowing named/markdown/all-caps headings through.
    in_terminal_boilerplate = False

    def flush() -> None:
        nonlocal buf, buf_start, buf_end
        if buf and buf_start is not None:
            body = "\n".join(buf).strip()
            if body:
                blocks.append(_Block(
                    section=cur_section,
                    subsection=cur_subsection,
                    char_start=buf_start,
                    char_end=buf_end,
                    text=body,
                ))
        buf = []
        buf_start = None

    # Walk lines while tracking original-text offsets. We can't use
    # text.splitlines() because it loses offsets; do it manually.
    pos = 0
    n = len(text)
    _terminal = frozenset({
        "references", "reference", "bibliography",
        "works cited", "literature cited",
    })
    while pos < n:
        nl = text.find("\n", pos)
        end = n if nl == -1 else nl
        line = text[pos:end]
        line_start = pos
        line_end = end + (1 if nl != -1 else 0)  # include trailing \n if any

        cls = _classify_line(line)
        # Suppress numbered-heading matches when we're inside a references
        # block — those are almost always citation entries, not headings.
        if cls is not None and in_terminal_boilerplate and cls[2] is not None:
            cls = None

        if cls is not None:
            level, title, _numbering = cls
            flush()
            if level == 1:
                cur_section = title
                cur_subsection = None
                in_terminal_boilerplate = (
                    title.strip().lower().rstrip(".:") in _terminal
                )
            elif level == 2:
                cur_subsection = title
            else:  # level 3 — treat as deeper subsection text, append to title
                if cur_subsection:
                    cur_subsection = f"{cur_subsection} / {title}"
                else:
                    cur_subsection = title
        else:
            # Body line. Skip blank lines at the start of a buffer to keep
            # offsets tight, but otherwise preserve internal blank lines.
            if not buf and not line.strip():
                pos = line_end
                continue
            if buf_start is None:
                buf_start = line_start
            buf.append(line)
            buf_end = end  # offset just past the last non-newline char
        pos = line_end

    flush()
    return blocks


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char offsets for each whitespace-separated word in
    `text`. Used to convert word-window boundaries back to character offsets
    in the original text.
    """
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _window_block(
    block: _Block,
    target_words: int,
    overlap_words: int,
) -> list[_Block]:
    """Split a block into overlapping word-windows, preserving section
    metadata and translating offsets back to the original text.
    """
    spans = _word_spans(block.text)
    if len(spans) <= target_words:
        return [block]

    if overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("overlap_words must be in [0, target_words)")

    out: list[_Block] = []
    start = 0
    while start < len(spans):
        end = min(start + target_words, len(spans))
        s_char = spans[start][0]
        e_char = spans[end - 1][1]
        out.append(_Block(
            section=block.section,
            subsection=block.subsection,
            char_start=block.char_start + s_char,
            char_end=block.char_start + e_char,
            text=block.text[s_char:e_char],
        ))
        if end == len(spans):
            break
        start = end - overlap_words
    return out


def _merge_small_blocks(
    blocks: list[_Block],
    min_words: int,
    max_words: int,
) -> list[_Block]:
    """Coalesce adjacent blocks that share the same section when the
    accumulator is below `min_words`, capping at `max_words`. Prevents a
    rash of tiny chunks for papers where sections are short.
    """
    if not blocks:
        return blocks

    out: list[_Block] = []
    for b in blocks:
        if not out:
            out.append(b)
            continue
        prev = out[-1]
        prev_wc = len(prev.text.split())
        b_wc = len(b.text.split())
        same_section = prev.section == b.section and prev.subsection == b.subsection
        if prev_wc < min_words and same_section and prev_wc + b_wc <= max_words:
            # Merge by stitching original text via char offsets if contiguous,
            # otherwise just join with a blank line.
            joined_text = prev.text + "\n\n" + b.text
            out[-1] = _Block(
                section=prev.section,
                subsection=prev.subsection,
                char_start=prev.char_start,
                char_end=b.char_end,
                text=joined_text,
            )
        else:
            out.append(b)
    return out


# ---------------------------------------------------------------------------
# Public helpers (kept for compatibility with existing callers / notebooks)
# ---------------------------------------------------------------------------


def chunk_by_section(text: str) -> list[str]:
    """Split text on detected section boundaries. Returns a list of strings,
    one per section. Empty input -> empty list. No detected headings -> a
    single-element list containing the original text.

    Preserved for backward compatibility; new code should prefer
    `chunk_paper`, which returns full Chunk metadata.
    """
    blocks = _parse_blocks(text)
    if not blocks:
        return [text] if text else []
    if len(blocks) == 1 and blocks[0].section is None:
        return [text]
    parts: list[str] = []
    for b in blocks:
        header = b.section or ""
        if b.subsection:
            header = f"{header} — {b.subsection}" if header else b.subsection
        parts.append(f"{header}\n{b.text}".strip() if header else b.text)
    return parts


def chunk_fixed_window(
    text: str,
    window_size: int = DEFAULT_TARGET_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split text into overlapping fixed-size word windows."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be in [0, window_size)")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + window_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def chunk_paper(
    text: str,
    paper_id: str | None = None,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    min_words: int = DEFAULT_MIN_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    drop_boilerplate: bool = True,
) -> list[Chunk]:
    """Chunk a research paper into embedding-ready pieces.

    Strategy:
        1. Parse the text into section-keyed blocks using a heading detector
           that handles markdown (`## Foo`), numbered (`3.1 Foo`), named
           (`Introduction`), and ALL-CAPS headings.
        2. Optionally drop reference and acknowledgement boilerplate.
        3. Merge tiny adjacent same-section blocks up to `min_words`.
        4. Split any block over `max_words` into overlapping word-windows of
           ~`target_words`, with `overlap_words` of overlap.
        5. If no headings were found at all, fall back to a flat sliding
           window over the whole document.

    Args:
        text: The paper text. Plain text or markdown both work.
        paper_id: Optional identifier propagated to every chunk.
        target_words: Desired chunk size when windowing.
        max_words: Hard upper bound; blocks longer than this are windowed.
        min_words: Adjacent same-section blocks below this are merged.
        overlap_words: Word overlap between windows of an oversized block.
        drop_boilerplate: If True, drop References, Acknowledgements, etc.

    Returns:
        A list of `Chunk` objects in document order, with `chunk_index`
        assigned sequentially starting at 0.
    """
    if not text or not text.strip():
        return []
    if target_words <= 0 or max_words < target_words or min_words < 0:
        raise ValueError("invalid sizing parameters")

    blocks = _parse_blocks(text)

    # If parsing didn't pick up any headings, blocks will be a single block
    # with section=None covering the whole document. Use the flat window
    # fallback in that case so callers still get reasonable chunks.
    no_headings = (
        not blocks
        or (len(blocks) == 1 and blocks[0].section is None and blocks[0].subsection is None)
    )
    if no_headings:
        return _flat_window_chunks(
            text,
            paper_id=paper_id,
            target_words=target_words,
            overlap_words=overlap_words,
        )

    # Drop boilerplate (refs, ack, funding, etc.) if requested.
    if drop_boilerplate:
        blocks = [b for b in blocks if not _block_is_boilerplate(b)]

    blocks = _merge_small_blocks(blocks, min_words=min_words, max_words=max_words)

    sized: list[_Block] = []
    for b in blocks:
        if len(b.text.split()) > max_words:
            sized.extend(_window_block(b, target_words, overlap_words))
        else:
            sized.append(b)

    chunks: list[Chunk] = []
    for i, b in enumerate(sized):
        chunks.append(Chunk(
            content=b.text,
            section=b.section,
            subsection=b.subsection,
            chunk_index=i,
            char_start=b.char_start,
            char_end=b.char_end,
            paper_id=paper_id,
        ))
    return chunks


def _block_is_boilerplate(block: _Block) -> bool:
    section = (block.section or "").strip().lower().rstrip(".:")
    subsection = (block.subsection or "").strip().lower().rstrip(".:")
    return section in _DROP_SECTIONS or subsection in _DROP_SECTIONS


def chunk_paper_structured(
    structured,  # research_rag.ingest.StructuredPdf
    paper_id: str | None = None,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    min_words: int = DEFAULT_MIN_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    drop_boilerplate: bool = True,
    size_offset: float = 0.5,
) -> list[Chunk]:
    """Chunk from a font-aware StructuredPdf.

    Renders the structured representation to text via to_markdown(),
    which prefixes ``## `` onto every line the font analysis flagged as
    a heading (larger size or bold variant of the body font). The
    existing chunk_paper machinery then picks those markers up via its
    markdown branch — far more reliable than re-running the text-only
    heading heuristics on extracted PDF output, particularly for papers
    where the body font and section heading font are identical except
    for weight (e.g. LCS 2024 with URWPalladioL-Roma vs -Bold).

    All other knobs (target_words, drop_boilerplate, etc.) are forwarded
    to chunk_paper unchanged.
    """
    text = structured.to_markdown(size_offset=size_offset)
    return chunk_paper(
        text,
        paper_id=paper_id,
        target_words=target_words,
        max_words=max_words,
        min_words=min_words,
        overlap_words=overlap_words,
        drop_boilerplate=drop_boilerplate,
    )


def _flat_window_chunks(
    text: str,
    *,
    paper_id: str | None,
    target_words: int,
    overlap_words: int,
) -> list[Chunk]:
    """Sliding-window chunking with accurate char offsets, used when no
    headings were detected.
    """
    spans = _word_spans(text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(spans):
        end = min(start + target_words, len(spans))
        s_char = spans[start][0]
        e_char = spans[end - 1][1]
        chunks.append(Chunk(
            content=text[s_char:e_char],
            section=None,
            subsection=None,
            chunk_index=idx,
            char_start=s_char,
            char_end=e_char,
            paper_id=paper_id,
        ))
        idx += 1
        if end == len(spans):
            break
        start = end - overlap_words
    return chunks
