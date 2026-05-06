"""Tests for PromptTemplate parsing/render and PromptRegistry."""
from __future__ import annotations

import pytest

from research_rag import PromptRegistry, PromptTemplate
from research_rag.config import PROMPTS_DIR


def test_prompttemplate_parses_real_summarization():
    p = PromptTemplate.from_file(PROMPTS_DIR / "summarization" / "v1.md")
    assert p.name == "summarization"
    assert p.version == "v1"
    assert "{paper_text}" in p.user_template
    assert p.system  # non-empty


def test_prompttemplate_qa_is_system_only():
    p = PromptTemplate.from_file(PROMPTS_DIR / "qa" / "v1.md")
    assert p.user_template == ""
    assert p.system


def test_render_substitutes_known_keys_and_preserves_json_braces():
    p = PromptTemplate.from_file(PROMPTS_DIR / "grading" / "v1.md")
    _, user = p.render(question="Q?", answer="A.", ground_truth="GT", retrieved_context="CTX")
    assert "Q?" in user and "GT" in user and "{question}" not in user
    # The literal JSON example in the prompt body should remain intact.
    assert '"faithfulness"' in user


def test_render_leaves_unknown_braces_alone():
    p = PromptTemplate(
        name="demo", version="v1",
        system="sys", user_template="hello {known} and {unknown}",
        metadata={},
    )
    _, user = p.render(known="WORLD")
    assert user == "hello WORLD and {unknown}"


def test_registry_lists_real_prompts():
    reg = PromptRegistry()
    names = reg.list_names()
    assert {"qa", "grading", "summarization"}.issubset(names)
    for name in names:
        assert reg.list_versions(name)


def test_registry_save_round_trip(tmp_path):
    reg = PromptRegistry(root=tmp_path)
    p = PromptTemplate(
        name="demo", version="v3",
        system="hello", user_template="say {x}",
        metadata={"name": "demo", "version": "v3", "notes": "multi-line\nblock"},
    )
    reg.save(p)
    p2 = reg.load("demo", "v3")
    assert p2.system == p.system
    assert p2.user_template == p.user_template
    assert p2.metadata.get("notes") == "multi-line\nblock"


def test_registry_latest_resolves_numerically(tmp_path):
    reg = PromptRegistry(root=tmp_path)
    for v in ("v1", "v2", "v10"):
        reg.save(PromptTemplate(
            name="task", version=v, system="s", user_template="",
            metadata={"name": "task", "version": v},
        ))
    assert reg._latest_version("task") == "v10"


def test_registry_load_nonexistent_raises(tmp_path):
    reg = PromptRegistry(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        reg.load("nonexistent", "latest")
