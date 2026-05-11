"""PDF -> TEI XML extraction via a local GROBID server.

GROBID is launched separately (see ``grobid_docker.GrobidDocker``); this
module just speaks HTTP to it. The TEI XML it returns has clean section
boundaries (``<div><head>``) and is what ``chunking.chunk_paper`` consumes
directly. ``tei_to_markdown`` is a small renderer used to drop a
human-readable sidecar into ``cache/parsed/``.

Public surface:
    PaperIngestionError - raised on any extraction failure
    parse_pdf(path)     - PDF -> TEI XML string (the canonical entry point)
    tei_to_markdown(xml)- TEI XML -> markdown (sidecar artifact)
    hash_pdf(path)      - sha256 prefix used as paper_id
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PaperIngestionError(Exception):
    """Raised when a PDF cannot be parsed."""


def hash_pdf(pdf_path: Path) -> str:
    """Return a 16-char prefix of sha256(pdf_bytes), used as paper_id."""
    pdf_path = Path(pdf_path)
    h = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _grobid_url() -> str:
    return os.environ.get("RAG_GROBID_URL", "http://localhost:8070")


def parse_pdf(
    pdf_path: Path,
    *,
    grobid_url: str | None = None,
    timeout: float = 180.0,
) -> str:
    """POST a PDF to GROBID and return the TEI XML body."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise PaperIngestionError(f"File not found: {pdf_path}")

    base = grobid_url or _grobid_url()
    endpoint = f"{base.rstrip('/')}/api/processFulltextDocument"

    try:
        with pdf_path.open("rb") as f:
            resp = requests.post(
                endpoint,
                files={"input": (pdf_path.name, f, "application/pdf")},
                data={"consolidateHeader": "1", "segmentSentences": "0"},
                timeout=timeout,
            )
    except requests.RequestException as e:
        raise PaperIngestionError(
            f"GROBID request failed for {pdf_path.name} ({endpoint}): {e}"
        ) from e

    if resp.status_code != 200:
        raise PaperIngestionError(
            f"GROBID returned {resp.status_code} for {pdf_path.name}: "
            f"{resp.text[:200]}"
        )
    if not resp.text.strip():
        raise PaperIngestionError(f"GROBID returned empty body for {pdf_path.name}")
    return resp.text


_BACK_MATTER_HEADS = frozenset({
    "references", "reference", "bibliography", "works cited", "literature cited",
    "acknowledgements", "acknowledgments", "acknowledgement", "acknowledgment",
    "funding", "author contributions", "conflicts of interest",
    "conflict of interest", "competing interests", "disclosure", "disclosures",
    "data availability", "data availability statement",
    "supplementary material", "supplementary materials", "supporting information",
})


def tei_to_markdown(tei_xml: str) -> str:
    """Render a TEI document as markdown for the human-readable sidecar.

    Title, abstract, and each top-level body ``<div>``'s head + paragraphs.
    Back-matter (references, acknowledgements) is skipped.
    """
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    parts: list[str] = []

    title_el = soup.select_one("teiHeader titleStmt title")
    if title_el and title_el.get_text(strip=True):
        parts.append(f"# {title_el.get_text(' ', strip=True)}")

    abstract = soup.select_one("teiHeader profileDesc abstract")
    if abstract:
        abs_paras = [p.get_text(" ", strip=True) for p in abstract.find_all("p")]
        abs_paras = [p for p in abs_paras if p]
        if abs_paras:
            parts.append("## Abstract")
            parts.extend(abs_paras)

    body = soup.select_one("text body")
    if body:
        for div in body.find_all("div", recursive=False):
            _render_div(div, parts, level=2)

    return "\n\n".join(parts).strip() + "\n"


def _render_div(div, parts: list[str], level: int) -> None:
    head = div.find("head", recursive=False)
    head_text = head.get_text(" ", strip=True) if head else ""
    if head_text and head_text.lower().rstrip(".:") in _BACK_MATTER_HEADS:
        return
    if head_text:
        parts.append(f"{'#' * min(level, 6)} {head_text}")
    for child in div.find_all(["p", "div"], recursive=False):
        if child.name == "p":
            text = child.get_text(" ", strip=True)
            if text:
                parts.append(text)
        else:
            _render_div(child, parts, level + 1)
