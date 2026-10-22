"""Retrieval: find the knowledge-base chunks that best answer a question.

This is a from-scratch BM25 ranker (the classic search-engine scoring formula)
with light stemming and a small synonym table for student vocabulary
("drop a class" ~ "course", "tuition" ~ "pay"). It needs no database and no
embedding model, so the whole system runs offline for tests and demos.

The `Retriever` interface is deliberately small (`search(query, k)`), so a
PostgreSQL + pgvector semantic retriever can be added later and combined with
this one (hybrid search) without touching the agent core.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.kb.loader import Chunk

STOPWORDS = set(
    """a about after all also am an and any are as at be been before being but by can could
    did do does doing for from get got had has have how i if in into is it its just me my
    no not of on or our please should so than that the their them then there these they
    this to too up us was we were what when where which who why will with would you your
    want need know tell help hi hello thanks thank nau university go going make take many stay""".split()
)

# Map common student words to the words the official pages use.
SYNONYMS = {
    "class": "course", "classes": "course", "courses": "course",
    "pay": "payment", "paying": "payment", "fee": "cost", "fees": "cost", "price": "cost",
    "much": "cost", "money": "payment",
    "quit": "withdraw", "withdrawal": "withdraw", "leave": "withdraw",
    "signup": "register", "enroll": "register", "enrol": "register", "registration": "register",
    "schedule": "schedule", "timetable": "schedule",
    "transcripts": "transcript", "record": "transcript",
    "login": "account", "log": "account", "password": "password", "email": "email",
    "job": "work", "jobs": "work", "working": "work", "employment": "work", "internship": "cpt",
    "visa": "f1",
    "tutor": "tutoring", "tutors": "tutoring",
    "disability": "accommodation", "disabilities": "accommodation", "accommodations": "accommodation",
    "loan": "loan", "loans": "loan", "grant": "grant", "grants": "grant", "scholarship": "scholarship",
    "refunds": "refund", "money back": "refund",
}

_TOKEN = re.compile(r"[a-z0-9]+")
_CODE = re.compile(r"\b([fij])-(\d+)\b")  # f-1 -> f1, i-20 -> i20


def stem(word: str) -> str:
    """Very small suffix stripper: 'registering' -> 'register', 'holds' -> 'hold'."""
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    return word


def tokenize(text: str) -> list[str]:
    tokens = []
    text = _CODE.sub(r"\1\2", text.lower())
    for raw in _TOKEN.findall(text):
        if raw in STOPWORDS:
            continue
        raw = SYNONYMS.get(raw, raw)
        tokens.append(stem(raw))
    return tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float      # BM25 score (higher is better)
    coverage: float   # IDF-weighted share of the question's distinct terms found in this chunk (0..1)
    matched: int = 0  # number of distinct question terms found in this chunk
    n_terms: int = 0  # number of distinct question terms

    @property
    def confident(self) -> bool:
        """Enough of the question is covered to answer from this chunk.

        At least two distinct question terms must match (or every term, for
        one-word questions). A single shared word like "time" is not enough
        evidence that a source answers the question.
        """
        return self.matched >= min(2, self.n_terms)


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Index the heading twice so section titles weigh more than body text.
        self.docs = [tokenize(f"{c.section} {c.section} {c.title} {c.text}") for c in chunks]
        self.tfs = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[Hit]:
        terms = list(dict.fromkeys(tokenize(query)))  # distinct, order kept
        if not terms:
            return []
        hits: list[Hit] = []
        for chunk, doc, tf in zip(self.chunks, self.docs, self.tfs):
            score = 0.0
            matched_weight = 0.0
            matched = 0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
                matched_weight += idf
                matched += 1
            if score > 0:
                # Words the knowledge base has never seen count at full weight: they are
                # usually the subject of the question ("gym", "parking").
                total_weight = sum(self.idf.get(t, max(self.idf.values(), default=1.0)) for t in terms)
                hits.append(Hit(chunk, score, matched_weight / total_weight if total_weight else 0.0,
                                matched, len(terms)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
