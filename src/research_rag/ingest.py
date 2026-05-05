"""PDF -> text extraction. Hierarchy: pypdf -> pdfplumber -> optional OCR."""
from __future__ import annotations

import hashlib
import logging
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

    Tries pypdf first; on empty/short output or exception, falls back to
    pdfplumber. OCR via pytesseract only runs when explicitly requested.
    Raises PaperIngestionError naming the file when all parsers fail.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise PaperIngestionError(f"File not found: {pdf_path}")

    pypdf_error: Exception | None = None
    try:
        text = _extract_with_pypdf(pdf_path)
        if not _looks_empty(text):
            return text
        logger.info("pypdf produced near-empty output for %s; trying pdfplumber", pdf_path.name)
    except Exception as e:
        pypdf_error = e
        logger.warning("pypdf failed on %s: %s", pdf_path.name, e)

    plumber_error: Exception | None = None
    try:
        text = _extract_with_pdfplumber(pdf_path)
        if not _looks_empty(text):
            return text
        logger.info("pdfplumber produced near-empty output for %s", pdf_path.name)
    except Exception as e:
        plumber_error = e
        logger.warning("pdfplumber failed on %s: %s", pdf_path.name, e)

    if ocr:
        text = _ocr_pdf(pdf_path)
        if not _looks_empty(text):
            return text

    raise PaperIngestionError(
        f"All parsers failed for {pdf_path.name}. "
        f"pypdf: {pypdf_error!r}, pdfplumber: {plumber_error!r}. "
        f"Try re-running with ocr=True if the PDF is scanned."
    )
