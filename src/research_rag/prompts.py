"""PromptTemplate + PromptRegistry — markdown files with YAML frontmatter."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import PROMPTS_DIR


@dataclass
class PromptTemplate:
    """A versioned prompt loaded from a markdown file with YAML frontmatter.

    File format:
        ---
        name: <name>
        version: <version>
        ...other metadata
        ---

        # System
        <system prompt body>

        # User
        <user prompt body, with `{var}` placeholders for runtime substitution>

    Substitution only replaces `{identifier}` patterns where `identifier`
    matches a passed kwarg — literal `{` and `}` (e.g. JSON examples in
    the prompt body) are left untouched.
    """

    name: str
    version: str
    system: str
    user_template: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """Substitute kwargs into user_template; return (system, user)."""
        return self.system, _substitute(self.user_template, kwargs)

    @classmethod
    def from_file(cls, path: Path) -> PromptTemplate:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        meta = yaml.safe_load(frontmatter) or {}
        system, user = _split_system_user(body)
        return cls(
            name=str(meta.get("name", path.stem)),
            version=str(meta.get("version", path.stem)),
            system=system,
            user_template=user,
            metadata=meta,
        )

    def to_string(self) -> str:
        """Render back to YAML-frontmatter markdown form.

        Multi-line strings (e.g. notes:) emit with the literal block
        scalar style (|) so they stay readable in git diffs.
        """
        fm = yaml.dump(
            self.metadata,
            Dumper=_LiteralDumper,
            default_flow_style=False,
            sort_keys=False,
        ).strip()
        parts = ["---", fm, "---", "", "# System", "", self.system]
        if self.user_template:
            parts += ["", "# User", "", self.user_template]
        return "\n".join(parts) + "\n"


class PromptRegistry:
    """Filesystem-backed prompt registry. Reads/writes `prompts/{name}/{version}.md`."""

    def __init__(self, root: Path = PROMPTS_DIR):
        self.root = Path(root)

    def load(self, name: str, version: str = "latest") -> PromptTemplate:
        if version == "latest":
            version = self._latest_version(name)
        path = self.root / name / f"{version}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        return PromptTemplate.from_file(path)

    def list_versions(self, name: str) -> list[str]:
        d = self.root / name
        if not d.is_dir():
            return []
        return _sort_versions(p.stem for p in d.glob("*.md"))

    def list_names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def save(self, template: PromptTemplate) -> None:
        path = self.root / template.name / f"{template.version}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.to_string(), encoding="utf-8")

    def _latest_version(self, name: str) -> str:
        versions = self.list_versions(name)
        if not versions:
            raise FileNotFoundError(f"No prompt versions found under {self.root / name}")
        return versions[-1]


# ---- helpers ----

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _substitute(template: str, values: dict[str, Any]) -> str:
    """Replace `{key}` with str(values[key]) for every known key.

    Unknown placeholders and non-identifier braces are left as-is, so JSON
    examples in the prompt body don't get mangled.
    """

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(values[key]) if key in values else match.group(0)

    return _PLACEHOLDER_RE.sub(replace, template)


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a YAML-frontmatter markdown file into (frontmatter, body)."""
    if not (text.startswith("---\n") or text.startswith("---\r\n")):
        raise ValueError("Prompt file must begin with YAML frontmatter (---)")
    start = 4 if text.startswith("---\n") else 5
    rest = text[start:]
    end = rest.find("\n---")
    if end == -1:
        raise ValueError("Prompt file frontmatter not terminated by ---")
    return rest[:end], rest[end + 4 :].lstrip("\n")


_HEADER_RE = re.compile(r"(?m)^# (System|User)\s*$")


def _split_system_user(body: str) -> tuple[str, str]:
    """Split body on '# System' and '# User' headers.

    A prompt may have only '# System' (e.g. an interactive QA prompt
    where user messages come from the runtime caller); in that case the
    user template is the empty string.
    """
    headers = list(_HEADER_RE.finditer(body))
    if not headers:
        raise ValueError("Prompt body must contain at least a '# System' header")
    sections: dict[str, str] = {}
    for i, m in enumerate(headers):
        name = m.group(1).lower()
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        sections[name] = body[start:end].strip()
    if "system" not in sections:
        raise ValueError("Prompt body must contain a '# System' header")
    return sections["system"], sections.get("user", "")


_VERSION_RE = re.compile(r"^v(\d+)")


def _sort_versions(versions: Iterable[str]) -> list[str]:
    """Sort 'v1', 'v2', 'v10' numerically; fall back to lexicographic otherwise."""

    def key(v: str) -> tuple[int, int, str]:
        m = _VERSION_RE.match(v)
        if m:
            return (0, int(m.group(1)), v)
        return (1, 0, v)

    return sorted(versions, key=key)


class _LiteralDumper(yaml.SafeDumper):
    """SafeDumper that emits multi-line strings using the literal block style (|)."""


def _str_presenter(dumper: yaml.SafeDumper, data: str) -> Any:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralDumper.add_representer(str, _str_presenter)
