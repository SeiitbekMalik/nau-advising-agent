"""Answer generation behind a thin adapter.

Three interchangeable generators share one method, `answer(question, hits)`:

- ExtractiveGenerator: offline, no API key. Picks the most relevant sentences
  from the retrieved official text and cites them. Always grounded, never
  invents anything; used for tests, demos and as a safety fallback.
- AnthropicGenerator / OpenAIGenerator: call an LLM with ONLY the retrieved
  sources and strict instructions to answer from them and cite [n].

If an LLM call fails, or returns an answer without a valid citation, the
agent falls back to the extractive answer, so a reply is always grounded.
"""

from __future__ import annotations

import re
from typing import Protocol

import httpx

from app.kb.retriever import Hit, tokenize

SYSTEM_PROMPT = """You are NAU's automated student advising assistant for North American University (Houston, TX).
Rules:
1. Answer ONLY with facts from the numbered SOURCES. Never use outside knowledge, never guess dates, fees or rules.
2. After every factual sentence, cite its source like [1] or [2].
3. If the sources do not contain the answer, reply exactly: NO_ANSWER
4. You cannot see any student's personal records (grades, balance, holds, status). If asked, explain where the student can check.
5. Never make decisions about admissions, academic standing, financial aid awards, discipline or immigration status.
6. Be brief and friendly: 2-5 sentences, plain language, include the relevant contact or link from the sources."""


class Generator(Protocol):
    name: str

    def answer(self, question: str, hits: list[Hit]) -> str: ...


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {h.chunk.label} ({h.chunk.source})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    )


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


class ExtractiveGenerator:
    """Offline answerer: returns the best-matching sentences of the top source(s)."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def answer(self, question: str, hits: list[Hit]) -> str:
        q_terms = set(tokenize(question))
        top = hits[0].score
        # Use the best source, plus any runner-up that scored almost as well
        # (e.g. "How do I pay tuition?" matches both "How to pay" and "Payment plan").
        candidates: list[tuple[int, int, int, str]] = []  # (overlap, -source, -position, sentence)
        for n, hit in enumerate(hits, start=1):
            if hit.score < 0.75 * top or not hit.confident:
                continue
            for pos, s in enumerate(_SENTENCE.split(hit.chunk.text)):
                candidates.append((len(q_terms & set(tokenize(s))), -n, -pos, s))
        best = sorted(candidates, reverse=True)[: self.max_sentences]
        best = [c for c in best if c[0] > 0] or best[:1]
        # Show the chosen sentences grouped by source, in their original order, each source cited.
        parts = []
        for n in sorted({-c[1] for c in best}):
            sents = sorted((c for c in best if -c[1] == n), key=lambda c: -c[2])
            parts.append(" ".join(c[3] for c in sents) + f" [{n}]")
        return " ".join(parts)


class _HTTPGenerator:
    name = "llm"
    timeout = 30.0

    def _user_prompt(self, question: str, hits: list[Hit]) -> str:
        return f"SOURCES:\n{format_sources(hits)}\n\nSTUDENT QUESTION: {question}"


class AnthropicGenerator(_HTTPGenerator):
    name = "anthropic"

    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "claude-haiku-5-5"

    def answer(self, question: str, hits: list[Hit]) -> str:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": self.model,
                "max_tokens": 400,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": self._user_prompt(question, hits)}],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip()


class OpenAIGenerator(_HTTPGenerator):
    name = "openai"

    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"

    def answer(self, question: str, hits: list[Hit]) -> str:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": 400,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._user_prompt(question, hits)},
                ],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def build_generator(provider: str, anthropic_key: str = "", openai_key: str = "", model: str = "") -> Generator:
    if provider == "anthropic" and anthropic_key:
        return AnthropicGenerator(anthropic_key, model)
    if provider == "openai" and openai_key:
        return OpenAIGenerator(openai_key, model)
    return ExtractiveGenerator()


CITATION = re.compile(r"\[(\d+)\]")


def cited_indices(text: str, n_sources: int) -> list[int]:
    """Valid citation numbers in an answer, in order of first appearance."""
    seen: list[int] = []
    for m in CITATION.finditer(text):
        i = int(m.group(1))
        if 1 <= i <= n_sources and i not in seen:
            seen.append(i)
    return seen
