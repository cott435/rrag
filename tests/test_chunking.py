"""Tests for the TEI-based chunker."""
from __future__ import annotations

from research_rag import chunk_paper


def _tei(body_divs: str, abstract: str | None = None) -> str:
    abs_block = (
        f"<profileDesc><abstract><p>{abstract}</p></abstract></profileDesc>"
        if abstract else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc><titleStmt><title>Paper</title></titleStmt></fileDesc>{abs_block}</teiHeader>
  <text><body>{body_divs}</body></text>
</TEI>"""


def test_body_divs_yield_section_chunks():
    xml = _tei(
        "<div><head>Methods</head><p>We did things.</p></div>"
        "<div><head>Results</head><p>It worked.</p></div>"
    )
    chunks = chunk_paper(xml, paper_id="p1")
    sections = [c.section for c in chunks]
    assert "Methods" in sections and "Results" in sections


def test_abstract_emitted_as_chunk():
    xml = _tei(
        "<div><head>Intro</head><p>Body.</p></div>",
        abstract="Short abstract.",
    )
    chunks = chunk_paper(xml)
    assert any(c.section == "Abstract" and "Short abstract" in c.content for c in chunks)


def test_references_section_dropped_by_default():
    xml = _tei(
        "<div><head>Methods</head><p>Methods.</p></div>"
        "<div><head>References</head><p>[1] Smith et al.</p></div>"
    )
    chunks = chunk_paper(xml)
    sections = [(c.section or "").lower() for c in chunks]
    assert "references" not in sections


def test_references_kept_when_drop_boilerplate_false():
    xml = _tei(
        "<div><head>Methods</head><p>Methods.</p></div>"
        "<div><head>References</head><p>[1] Smith et al.</p></div>"
    )
    chunks = chunk_paper(xml, drop_boilerplate=False)
    sections = [(c.section or "").lower() for c in chunks]
    assert "references" in sections


def test_long_section_split_into_windows():
    long_text = " ".join(["word"] * 1500)
    xml = _tei(f"<div><head>Discussion</head><p>{long_text}</p></div>")
    chunks = chunk_paper(xml, target_words=500, max_words=600, overlap_words=50)
    discussion = [c for c in chunks if c.section == "Discussion"]
    assert len(discussion) >= 2


def test_paper_id_propagates_to_chunks():
    xml = _tei("<div><head>Methods</head><p>We did things.</p></div>")
    chunks = chunk_paper(xml, paper_id="abc123")
    assert chunks and all(c.paper_id == "abc123" for c in chunks)


def test_nested_div_becomes_subsection():
    xml = _tei(
        "<div><head>Methods</head><p>Top body.</p>"
        "<div><head>Dataset</head><p>Inner body.</p></div></div>"
    )
    chunks = chunk_paper(xml)
    inner = [c for c in chunks if c.subsection == "Dataset"]
    assert inner and inner[0].section == "Methods"


def test_empty_xml_returns_empty():
    assert chunk_paper("") == []
    assert chunk_paper("   ") == []
