"""PDF -> text extraction. Hierarchy: pypdf -> pdfplumber -> optional OCR."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PaperIngestionError(Exception):
    """Raised when a PDF cannot be parsed by any available method."""


def hash_pdf(pdf_path: Path) -> str:
    """Return a 16-char prefix of sha256(pdf_bytes), used as paper_id."""
    pdf_path = Path(pdf_path)
    h = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


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
    """Treat <100 chars as a parser failure worth retrying with a fallback."""
    return len(text.strip()) < 100


# Matches the /CNN tokens that pypdf/pdfplumber emit when a font's ToUnicode
# CMap is missing or broken — the extractor falls back to raw glyph IDs from
# the font encoding (e.g. /C68/C101/C115 for "Des"). The text is unrecoverable
# without re-rasterizing and OCR'ing the page.
_GLYPH_ID_RE = re.compile(r"/C\d+")


def _looks_like_glyph_ids(text: str, threshold: float = 0.3) -> bool:
    """True if a substantial fraction of the output is /CNN glyph-ID tokens.

    Uses a fraction rather than presence-of-any so that legitimate text
    containing the literal substring "/C123" (rare, but possible in a code
    listing) doesn't trigger a false positive.
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
    """Best-effort OCR via pdf2image + pytesseract. Imports lazily so the base
    install doesn't require system-level poppler/tesseract.
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


def parse_pdf(pdf_path: Path, ocr: bool = False) -> str:
    """Extract text from a PDF, falling back across parsers as needed.

    Tries pypdf first; on empty/short/glyph-ID output or exception, falls back
    to pdfplumber. If both parsers produce glyph-ID gibberish (a sign of a
    broken ToUnicode CMap), OCR is triggered automatically regardless of the
    `ocr` flag, since the text is otherwise unrecoverable. For merely empty
    output (likely scanned pages), OCR runs only when `ocr=True`.

    Raises PaperIngestionError naming the file when all parsers fail.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise PaperIngestionError(f"File not found: {pdf_path}")

    saw_glyph_ids = False

    pypdf_error: Exception | None = None
    try:
        text = _extract_with_pypdf(pdf_path)
        if not _is_bad_output(text):
            return text
        if _looks_like_glyph_ids(text):
            saw_glyph_ids = True
            logger.info(
                "pypdf returned glyph-ID tokens for %s (broken ToUnicode CMap); trying pdfplumber",
                pdf_path.name,
            )
        else:
            logger.info("pypdf produced near-empty output for %s; trying pdfplumber", pdf_path.name)
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
            logger.info(
                "pdfplumber returned glyph-ID tokens for %s (broken ToUnicode CMap)",
                pdf_path.name,
            )
        else:
            logger.info("pdfplumber produced near-empty output for %s", pdf_path.name)
    except Exception as e:
        plumber_error = e
        logger.warning("pdfplumber failed on %s: %s", pdf_path.name, e)

    # Auto-trigger OCR on glyph-ID output: the underlying text stream is
    # unrecoverable, so honoring ocr=False here would just guarantee failure.
    should_ocr = ocr or saw_glyph_ids
    if should_ocr:
        if saw_glyph_ids and not ocr:
            logger.warning(
                "Forcing OCR for %s because text-layer extraction produced glyph IDs",
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