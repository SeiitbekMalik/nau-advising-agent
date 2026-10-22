"""Ingest official NAU pages into the knowledge base.

Usage:
    python -m app.kb.ingest                    # fetch every URL in data/sources.txt
    python -m app.kb.ingest https://www.na.edu/library/   # fetch specific URLs

For each URL it downloads the page (HTML or PDF), keeps the main text, splits
it into sections at headings, and writes one markdown file with front matter
into data/kb/web/. The agent picks the files up on the next start.

Only public pages are ingested. Never point this at the student portal or any
page that holds personal records.
"""

from __future__ import annotations

import io
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import ROOT, settings

SOURCES_FILE = ROOT / "data" / "sources.txt"
OUT_DIR = settings.kb_dir / "web"
BLOCKED_HOSTS = {"portal.na.edu", "na.instructure.com", "login.microsoftonline.com"}
MAX_SECTION_WORDS = 220  # long sections are split so each chunk stays specific


def slug_for(url: str) -> str:
    path = urlparse(url).path.strip("/") or "home"
    return re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")[:80]


def _split_long(heading: str, text: str) -> list[tuple[str, str]]:
    words = text.split()
    if len(words) <= MAX_SECTION_WORDS:
        return [(heading, text)]
    parts = []
    for i in range(0, len(words), MAX_SECTION_WORDS):
        n = i // MAX_SECTION_WORDS + 1
        parts.append((f"{heading} (part {n})", " ".join(words[i:i + MAX_SECTION_WORDS])))
    return parts


def html_to_sections(html: str) -> tuple[str, list[tuple[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript", "iframe"]):
        tag.decompose()
    title = (soup.find("h1") or soup.title or soup).get_text(" ", strip=True)[:120] or "NAU page"
    body = soup.find("main") or soup.find("article") or soup.body or soup
    sections: list[tuple[str, str]] = []
    heading, buf = "Overview", []
    for el in body.find_all(["h2", "h3", "h4", "p", "li", "td"]):
        if el.name in {"h2", "h3", "h4"}:
            if buf:
                sections += _split_long(heading, " ".join(buf))
            heading, buf = el.get_text(" ", strip=True) or heading, []
        else:
            text = el.get_text(" ", strip=True)
            if text and text not in buf:
                buf.append(text)
    if buf:
        sections += _split_long(heading, " ".join(buf))
    return title, sections


def pdf_to_sections(data: bytes, url: str) -> tuple[str, list[tuple[str, str]]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    title = Path(urlparse(url).path).stem.replace("-", " ").title()
    sections: list[tuple[str, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if len(text) > 80:
            sections += _split_long(f"Page {i}", text)
    return title, sections


def write_markdown(url: str, title: str, sections: list[tuple[str, str]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{slug_for(url)}.md"
    lines = ["---", f"title: {title}", f"source: {url}", f"retrieved: {date.today().isoformat()}", "---", ""]
    for heading, text in sections:
        lines += [f"## {heading}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def ingest(url: str, client: httpx.Client) -> Path | None:
    host = urlparse(url).hostname or ""
    if host in BLOCKED_HOSTS:
        print(f"skip (private system): {url}")
        return None
    r = client.get(url)
    r.raise_for_status()
    if "pdf" in r.headers.get("content-type", "") or url.lower().endswith(".pdf"):
        title, sections = pdf_to_sections(r.content, url)
    else:
        title, sections = html_to_sections(r.text)
    if not sections:
        print(f"no text found: {url}")
        return None
    path = write_markdown(url, title, sections)
    print(f"ok  {len(sections):3d} sections  {url}  ->  {path.relative_to(ROOT)}")
    return path


def main(argv: list[str]) -> int:
    urls = argv or [
        line.strip() for line in SOURCES_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    headers = {"User-Agent": "NAU-Advising-Agent-Capstone/0.1 (student project)"}
    failures = 0
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        for url in urls:
            try:
                ingest(url, client)
            except Exception as exc:
                failures += 1
                print(f"FAILED {url}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
