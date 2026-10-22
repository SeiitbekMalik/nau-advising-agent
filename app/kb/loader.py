"""Load the knowledge base: markdown files -> small, citable chunks.

Each file starts with a front-matter block (title, source URL, retrieved date),
and each "## " section becomes one chunk. One chunk = one answerable topic,
which keeps citations precise ("Registrar's Office > How to add or drop a class").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str           # e.g. "registrar#how-to-add-or-drop-a-class"
    title: str        # document title, e.g. "Registrar's Office"
    section: str      # section heading
    text: str         # section body
    source: str       # official URL the content came from
    retrieved: str    # date the page was collected

    @property
    def label(self) -> str:
        return f"{self.title} > {self.section}"


_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_markdown(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    match = _FRONT_MATTER.match(raw)
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        raw = raw[match.end():]

    title = meta.get("title", path.stem.replace("-", " ").title())
    source = meta.get("source", "")
    retrieved = meta.get("retrieved", "")

    chunks: list[Chunk] = []
    # Split on level-2 headings; text before the first heading is ignored if empty.
    parts = re.split(r"^##\s+(.+)$", raw, flags=re.M)
    intro = parts[0].strip()
    if intro:
        chunks.append(Chunk(f"{path.stem}#overview", title, "Overview", intro, source, retrieved))
    for heading, body in zip(parts[1::2], parts[2::2]):
        body = " ".join(body.split())
        if body:
            chunks.append(Chunk(f"{path.stem}#{_slug(heading)}", title, heading.strip(), body, source, retrieved))
    return chunks


def load_kb(kb_dir: Path) -> list[Chunk]:
    """Load every .md file under kb_dir (including the web/ sub-folder made by ingestion)."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.rglob("*.md")):
        chunks.extend(parse_markdown(path))
    return chunks
