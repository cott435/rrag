"""PDF -> text extraction with font-aware heading detection.

Pipeline:
    1. Try the font-aware path: pdfplumber's char-level metadata is grouped
       into PdfLines, body font/size is inferred, running headers/footers
       are filtered, and each line is rendered as plain text — with `## `
       markdown markers prefixed onto detected headings. Headings are
       identified by either a larger font size than body OR a bold-variant
       font name. The output feeds directly into chunk_paper, which uses
       the markdown markers as reliable section boundaries.
    2. If font-aware extraction fails or produces glyph-ID gibberish, fall
       back to plain pypdf, then plain pdfplumber.
    3. Auto-trigger OCR on glyph-ID output (broken ToUnicode CMap), since
       the underlying text stream is unrecoverable without re-rasterizing.

Public surface:
    PaperIngestionError       - raised when no parser succeeds
    PdfLine, StructuredPdf    - structured-representation dataclasses
    parse_pdf(path, ...)      - returns plain text (markdown-marked); main entry
    parse_pdf_structured(p)   - returns the structured representation
    hash_pdf(path)            - sha256 prefix used as paper_id
    is_bold_font(name)        - bold-variant detector exposed for tests
"""
from __future__ import annotations

import collections
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PaperIngestionError(Exception):
    """Raised when a PDF cannot be parsed by any available method."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def hash_pdf(pdf_path: Path) -> str:
    """Return a 16-char prefix of sha256(pdf_bytes), used as paper_id."""
    pdf_path = Path(pdf_path)
    h = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Font-name analysis
# ---------------------------------------------------------------------------

# Name fragments that signal a bold/heavy variant. Matched against a
# normalized font name (lowercased, with the random "ABCDEF+" pdfplumber
# subset prefix stripped). Headings frequently differ from body only by
# weight — e.g. LCS 2024 uses URWPalladioL-Bold for headings at the same
# point size as the URWPalladioL-Roma body — so this is critical signal.
_BOLD_FONT_TOKENS = (
    "bold",
    "black",
    "heavy",
    "demi",
    "medi",     # NimbusRomNo9L-Medi
    "semibold",
    "-bd",
    ".b",       # AdvOT...B style suffixes used by Adobe-encoded fonts
)


def is_bold_font(font_name: str) -> bool:
    """True if the font name contains a known bold marker.

    Strips the pdfplumber subset prefix (e.g. ``BTGPSN+`` in
    ``BTGPSN+NimbusRomNo9L-Medi``) before matching.
    """
    if not font_name:
        return False
    name = font_name.lower()
    if "+" in name:
        name = name.split("+", 1)[1]
    return any(tok in name for tok in _BOLD_FONT_TOKENS)


# ---------------------------------------------------------------------------
# Heading rendering — the function the unit tests target directly
# ---------------------------------------------------------------------------


def _format_lines_as_markdown(
    lines: Iterable[tuple[float, str]],
    *,
    body_size: float,
    size_offset: float = 0.5,
    max_heading_words: int = 20,
    min_heading_chars: int = 2,
) -> list[str]:
    """Render (size, text) lines as plain text with `## ` heading markers.

    A line is heading-eligible iff its size strictly exceeds
    ``body_size + size_offset``. Heading-eligible lines that are too
    long (more than ``max_heading_words`` words — likely bold sentences,
    not titles) or too short (fewer than ``min_heading_chars`` chars —
    likely glyph noise like dropped initials or footnote markers) are
    dropped or demoted accordingly.

    Adjacent heading lines merge into a single ``## `` block until a
    body line breaks the run — this glues together title fragments and
    multi-line section headers (e.g. "Three-Dimensional Structure of
    Novel Liver Cancer Biomarker / Liver Cancer-Specific...").
    """
    out: list[str] = []
    heading_buf: list[str] = []
    threshold = body_size + size_offset

    def flush_heading() -> None:
        if heading_buf:
            out.append("## " + " ".join(heading_buf))
            heading_buf.clear()

    for size, text in lines:
        text = text.strip()
        if not text:
            continue
        is_big = size > threshold
        too_long = len(text.split()) > max_heading_words
        too_short = len(text) < min_heading_chars
        if is_big and not too_long:
            if too_short:
                # Single-char heading-size lines are almost always noise
                # (drop caps, footnote markers, glyph fragments).
                continue
            heading_buf.append(text)
        else:
            flush_heading()
            out.append(text)
    flush_heading()
    return out


# ---------------------------------------------------------------------------
# Structured representation
# ---------------------------------------------------------------------------


@dataclass
class PdfLine:
    """One line of text extracted from a PDF, with font metadata."""

    text: str
    page_num: int           # 0-indexed
    y_top: float
    y_bot: float
    x_left: float
    x_right: float
    dominant_font: str      # full name including subset prefix
    dominant_size: float    # rounded to 1 decimal
    is_bold: bool
    is_running: bool = False  # repeating header/footer line


@dataclass
class StructuredPdf:
    """Font-aware structured representation of a PDF.

    `lines` are in reading order (with column-aware reordering on
    multi-column pages). `body_font` / `body_size` are the inferred
    most-common pairs across body pages (page 1 weighted lower, since
    titles distort the distribution).
    """

    lines: list[PdfLine]
    body_font: str
    body_size: float
    n_pages: int
    bold_size_bump: float = 2.0

    def to_markdown(self, size_offset: float = 0.5) -> str:
        """Render to plain text with ``## `` markers on detected headings.

        Bold non-body-font lines get a synthetic size bump before the
        size-based heading detector runs, so headings that are bold at
        body size (LCS 2024 case) are still picked up.
        """
        pairs: list[tuple[float, str]] = []
        for line in self.lines:
            if line.is_running:
                continue
            size = line.dominant_size
            if line.is_bold and line.dominant_font != self.body_font:
                size = max(size, self.body_size) + self.bold_size_bump
            pairs.append((size, line.text))
        rendered = _format_lines_as_markdown(
            pairs, body_size=self.body_size, size_offset=size_offset
        )
        return "\n".join(rendered)


# ---------------------------------------------------------------------------
# Page parsing — column detection, line aggregation
# ---------------------------------------------------------------------------


def _detect_n_cols(
    page,
    body_top_frac: float = 0.15,
    body_bot_frac: float = 0.15,
    gutter_ratio: float = 0.1,
) -> int:
    """Return 1 or 2 based on a vertical-gutter heuristic in the body region.

    Multi-column PDFs leave a near-empty vertical band near the page
    centerline that single-column pages do not. We bin char x-centers,
    restrict to the body region (cropping likely header/footer/figure
    bands), and check whether the middle 20% of bins is significantly
    sparser than the surrounding column regions.
    """
    chars = page.chars
    if not chars:
        return 1
    W, H = page.width, page.height
    body = [c for c in chars if body_top_frac * H < c["top"] < (1 - body_bot_frac) * H]
    if len(body) < 100:
        return 1
    bins = [0] * 100
    for ch in body:
        cx = (ch["x0"] + ch["x1"]) / 2
        b = min(99, max(0, int(cx / W * 100)))
        bins[b] += 1
    middle_min = min(bins[40:60])
    left_avg = sum(bins[20:40]) / 20
    right_avg = sum(bins[60:80]) / 20
    if left_avg > 0 and right_avg > 0:
        if middle_min < gutter_ratio * min(left_avg, right_avg):
            return 2
    return 1


def _text_line_to_pdf_line(tl: dict, page_num: int) -> PdfLine | None:
    """Convert a pdfplumber extract_text_lines() entry to a PdfLine."""
    text = (tl.get("text") or "").strip()
    if not text:
        return None
    chars = tl.get("chars") or []
    if not chars:
        return PdfLine(
            text=text,
            page_num=page_num,
            y_top=float(tl.get("top", 0.0)),
            y_bot=float(tl.get("bottom", 0.0)),
            x_left=float(tl.get("x0", 0.0)),
            x_right=float(tl.get("x1", 0.0)),
            dominant_font="",
            dominant_size=0.0,
            is_bold=False,
        )
    sizes = collections.Counter(round(float(c.get("size") or 0.0), 1) for c in chars)
    fonts = collections.Counter(c.get("fontname") or "" for c in chars)
    dom_size = sizes.most_common(1)[0][0]
    dom_font = fonts.most_common(1)[0][0]
    return PdfLine(
        text=text,
        page_num=page_num,
        y_top=float(tl.get("top", 0.0)),
        y_bot=float(tl.get("bottom", 0.0)),
        x_left=float(tl.get("x0", 0.0)),
        x_right=float(tl.get("x1", 0.0)),
        dominant_font=dom_font,
        dominant_size=dom_size,
        is_bold=is_bold_font(dom_font),
    )


def _lines_from_region(region, page_num: int) -> list[PdfLine]:
    try:
        text_lines = region.extract_text_lines(x_tolerance=0.5)
    except Exception as e:
        logger.warning("extract_text_lines failed on page %d region: %s", page_num, e)
        return []
    out: list[PdfLine] = []
    for tl in text_lines:
        ln = _text_line_to_pdf_line(tl, page_num)
        if ln is not None:
            out.append(ln)
    return out


def _extract_lines_for_page(page, page_num: int) -> list[PdfLine]:
    """Return reading-order lines for a page, splitting columns if needed."""
    n_cols = _detect_n_cols(page)
    if n_cols == 1:
        return _lines_from_region(page, page_num)

    # Multi-column: crop each column separately and concatenate.
    # Full-width content (e.g. a figure caption in the middle) gets split
    # between halves — its text becomes mildly jumbled but headings
    # remain detectable per side. The alternative (preserving full-width
    # content) costs heading reliability for body fidelity, which is the
    # wrong trade for retrieval.
    W = page.width
    H = page.height
    out: list[PdfLine] = []
    for i in range(n_cols):
        x_left = i * W / n_cols
        x_right = min((i + 1) * W / n_cols, W)
        try:
            col = page.crop((x_left, 0, x_right, H), relative=False)
        except Exception as e:
            logger.warning("crop failed on page %d col %d: %s", page_num, i, e)
            continue
        out.extend(_lines_from_region(col, page_num))
    return out


# ---------------------------------------------------------------------------
# Body font/size detection and running header/footer detection
# ---------------------------------------------------------------------------


def _detect_body_font_size(lines: list[PdfLine]) -> tuple[str, float]:
    """Char-weighted mode of font/size across body pages.

    Page 1 is excluded when the doc has multiple pages, since title and
    abstract use distorting sizes. Already-flagged running lines are
    excluded so headers/footers don't pollute the body distribution.
    """
    if not lines:
        return ("", 10.0)
    n_pages = max(l.page_num for l in lines) + 1
    body_lines = [
        l for l in lines
        if not l.is_running and (n_pages <= 2 or l.page_num >= 1)
    ]
    if not body_lines:
        body_lines = lines
    size_count: collections.Counter = collections.Counter()
    font_count: collections.Counter = collections.Counter()
    for line in body_lines:
        weight = max(1, len(line.text))
        size_count[line.dominant_size] += weight
        font_count[line.dominant_font] += weight
    body_size = size_count.most_common(1)[0][0] if size_count else 10.0
    body_font = font_count.most_common(1)[0][0] if font_count else ""
    return body_font, body_size


_RUNNING_DIGITS_RE = re.compile(r"\d+")


def _detect_running_lines(
    lines: list[PdfLine],
    n_pages: int,
    min_pages: int = 3,
) -> set[int]:
    """Indices of lines that look like repeating headers/footers.

    Groups lines by (digit-normalized text, y-band). A line is "running"
    if its group spans at least half of all pages. Digit normalization
    means "page 12 of 30" and "page 13 of 30" share a template.
    """
    if n_pages < min_pages:
        return set()
    groups: dict[tuple[str, int], list[int]] = {}
    for i, line in enumerate(lines):
        template = _RUNNING_DIGITS_RE.sub("#", line.text.lower())
        y_bucket = round(line.y_top / 20) * 20
        groups.setdefault((template, y_bucket), []).append(i)
    threshold = max(min_pages, n_pages // 2)
    running: set[int] = set()
    for indices in groups.values():
        unique_pages = {lines[i].page_num for i in indices}
        if len(unique_pages) >= threshold:
            running.update(indices)
    return running


# ---------------------------------------------------------------------------
# Plain-text fallbacks (preserve existing behavior)
# ---------------------------------------------------------------------------


def _extract_with_pypdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
    return "\n\n".join(parts)


def _looks_empty(text: str) -> bool:
    """<100 chars after stripping warrants a fallback parser."""
    return len(text.strip()) < 100


# Matches the glyph-ID tokens that pypdf/pdfplumber emit when a font's
# ToUnicode CMap is missing or broken — the extractor falls back to raw
# glyph IDs from the font encoding. Two forms in the wild: `/C68/C101/...`
# (pypdf) and `(cid:16)(cid:2)/...` (pdfplumber). Both are unrecoverable
# without re-rasterizing and OCR'ing the page.
_GLYPH_ID_RE = re.compile(r"/C\d+|\(cid:\d+\)")


def _looks_like_glyph_ids(text: str, threshold: float = 0.3) -> bool:
    """True if a substantial fraction of the output is /CNN glyph-ID tokens.

    Uses a fraction (rather than presence-of-any) so that legitimate text
    containing a literal "/C123" — rare but possible in code listings —
    doesn't trigger a false positive.
    """
    stripped = text.strip()
    if not stripped:
        return False
    matched_chars = sum(len(m) for m in _GLYPH_ID_RE.findall(stripped))
    return matched_chars / len(stripped) >= threshold


def _is_bad_output(text: str) -> bool:
    """Either too short or dominated by glyph-ID tokens — both warrant fallback."""
    return _looks_empty(text) or _looks_like_glyph_ids(text)


def _ocr_pdf(pdf_path: Path) -> str:
    """Best-effort OCR via pdf2image + pytesseract, imported lazily so the
    base install doesn't require system-level poppler/tesseract.
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except ImportError as e:
        raise PaperIngestionError(
            "OCR requested but pdf2image/pytesseract not installed. "
            "Install with: pip install -e '.[ocr]'"
        ) from e
    images = convert_from_path(str(pdf_path))
    return "\n\n".join(pytesseract.image_to_string(img) for img in images)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def parse_pdf_structured(pdf_path: Path) -> StructuredPdf:
    """Extract a font-aware structured representation of a PDF.

    Raises PaperIngestionError if pdfplumber returns nothing usable.
    Callers that just want plain text should use parse_pdf instead.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise PaperIngestionError(f"File not found: {pdf_path}")

    all_lines: list[PdfLine] = []
    n_pages = 0
    with pdfplumber.open(str(pdf_path)) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            try:
                page_lines = _extract_lines_for_page(page, i)
                all_lines.extend(page_lines)
            except Exception as e:
                logger.warning("page %d extraction failed for %s: %s", i, pdf_path.name, e)

    if not all_lines:
        raise PaperIngestionError(f"No text extracted from {pdf_path.name}")

    body_font, body_size = _detect_body_font_size(all_lines)
    running = _detect_running_lines(all_lines, n_pages)
    for i in running:
        all_lines[i].is_running = True

    return StructuredPdf(
        lines=all_lines,
        body_font=body_font,
        body_size=body_size,
        n_pages=n_pages,
    )


def parse_pdf(pdf_path: Path, ocr: bool = False, font_aware: bool = True) -> str:
    """Extract text from a PDF, falling back across parsers as needed.

    With font_aware=True (default), tries the structured pdfplumber path
    first and emits text with ``## `` markers on detected headings (via
    font size and weight). Falls through to plain pypdf if that yields
    no usable output, then plain pdfplumber. OCR auto-triggers on glyph-
    ID gibberish (broken ToUnicode CMap), since the underlying text
    stream is unrecoverable; for merely-empty output (likely scanned
    pages) OCR runs only when ocr=True is passed.

    Raises PaperIngestionError naming the file when all parsers fail.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise PaperIngestionError(f"File not found: {pdf_path}")

    saw_glyph_ids = False

    if font_aware:
        try:
            structured = parse_pdf_structured(pdf_path)
            text = structured.to_markdown()
            if not _is_bad_output(text):
                return text
            if _looks_like_glyph_ids(text):
                saw_glyph_ids = True
                logger.info(
                    "font-aware extraction returned glyph IDs for %s; trying pypdf",
                    pdf_path.name,
                )
            else:
                logger.info(
                    "font-aware extraction produced near-empty output for %s; trying pypdf",
                    pdf_path.name,
                )
        except PaperIngestionError:
            raise
        except Exception as e:
            logger.warning(
                "font-aware extraction failed for %s: %s; falling back",
                pdf_path.name, e,
            )

    pypdf_error: Exception | None = None
    try:
        text = _extract_with_pypdf(pdf_path)
        if not _is_bad_output(text):
            return text
        if _looks_like_glyph_ids(text):
            saw_glyph_ids = True
    except Exception as e:
        pypdf_error = e
        logger.warning("pypdf failed on %s: %s", pdf_path.name, e)

    plumber_error: Exception | None = None
    try:
        text = _extract_with_pdfplumber(pdf_path)
        if not _is_bad_output(text):
            return text
        if _looks_like_glyph_ids(text):
            saw_glyph_ids = True
    except Exception as e:
        plumber_error = e
        logger.warning("pdfplumber failed on %s: %s", pdf_path.name, e)

    # Auto-trigger OCR when glyph IDs were observed: the text stream is
    # unrecoverable from the PDF objects, so honoring ocr=False would just
    # guarantee failure. For merely-empty output, only OCR if requested.
    should_ocr = ocr or saw_glyph_ids
    if should_ocr:
        if saw_glyph_ids and not ocr:
            logger.warning(
                "Forcing OCR for %s (text-layer extraction produced glyph IDs)",
                pdf_path.name,
            )
        text = _ocr_pdf(pdf_path)
        if not _is_bad_output(text):
            return text

    raise PaperIngestionError(
        f"All parsers failed for {pdf_path.name}. "
        f"pypdf: {pypdf_error!r}, pdfplumber: {plumber_error!r}. "
        f"Try re-running with ocr=True if the PDF is scanned."
    )
