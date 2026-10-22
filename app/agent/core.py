"""The agent core: one function that turns a student message into a safe reply.

Pipeline (in this exact order):
  1. Escalation check   - sensitive topics go to a human, before anything else.
  2. Retrieval          - find official NAU text that matches the question.
  3. Confidence gate    - if the best source doesn't really cover the question,
                          say "I don't know" and offer a human (never guess).
  4. Generation         - LLM (or offline extractive mode) answers ONLY from the
                          retrieved sources and cites them [n].
  5. Citation check     - an answer without a valid citation is rejected and the
                          grounded extractive answer is used instead.
  6. Logging            - message, reply, citations and escalation are stored.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.agent import escalation
from app.agent.llm import ExtractiveGenerator, Generator, cited_indices
from app.kb.retriever import BM25Retriever, Hit
from app.storage import Store

log = logging.getLogger(__name__)

DISCLOSURE = (
    "Hi! I'm NAU's automated advising assistant (an AI, not a person). I answer from NAU's official "
    "pages and show my sources. For anything personal or sensitive I'll connect you with a human advisor."
)

NO_ANSWER = (
    "I couldn't find this in NAU's official information, so I don't want to guess. "
    "I've flagged your question for a human advisor. You can also ask the Registrar's Office "
    "(registrar@na.edu, (832) 230-5555)."
)


@dataclass
class Citation:
    n: int
    title: str
    section: str
    url: str
    retrieved: str


@dataclass
class AgentReply:
    reply: str
    escalate: bool
    category: str | None = None          # escalation category, if any
    handoff_to: str | None = None
    urgent: bool = False
    citations: list[Citation] = field(default_factory=list)
    generator: str = ""                   # which generator produced the text
    latency_ms: int = 0


class AdvisingAgent:
    def __init__(self, retriever: BM25Retriever, generator: Generator, store: Store,
                 top_k: int = 4, min_coverage: float = 0.25):
        self.retriever = retriever
        self.generator = generator
        self.fallback = ExtractiveGenerator()
        self.store = store
        self.top_k = top_k
        self.min_coverage = min_coverage

    def handle(self, student_id: str, message: str, channel: str = "web") -> AgentReply:
        start = time.perf_counter()
        first_contact = not self.store.history(student_id, limit=1)
        result = self._answer(message)
        if first_contact:
            # AI self-disclosure at the start of every new conversation.
            result.reply = f"{DISCLOSURE}\n\n{result.reply}"
        result.latency_ms = int((time.perf_counter() - start) * 1000)

        self.store.add_message(student_id, channel, "student", message)
        self.store.add_message(student_id, channel, "agent", result.reply,
                               citations=[c.__dict__ for c in result.citations],
                               escalated=result.escalate, category=result.category)
        if result.escalate:
            self.store.add_escalation(student_id, channel, result.category or "unknown",
                                      result.urgent, result.handoff_to or "Academic Advisor", message)
        return result

    # --- pipeline -------------------------------------------------------------------------
    def _answer(self, message: str) -> AgentReply:
        # 1. Escalation rules always run first.
        esc = escalation.check(message)
        if esc:
            return AgentReply(reply=esc.message, escalate=True, category=esc.category,
                              handoff_to=esc.handoff_to, urgent=esc.urgent, generator="escalation-rules")

        # 2. Retrieval.
        hits = self.retriever.search(message, k=self.top_k)

        # 3. Confidence gate: weak match -> don't guess, hand off.
        if not hits or not hits[0].confident or hits[0].coverage < self.min_coverage:
            return AgentReply(reply=NO_ANSWER, escalate=True, category="no_grounded_answer",
                              handoff_to="Academic Advisor", generator="confidence-gate")

        # 4. Generation, 5. citation check.
        text, used = self._generate(message, hits)
        if text is None:
            return AgentReply(reply=NO_ANSWER, escalate=True, category="no_grounded_answer",
                              handoff_to="Academic Advisor", generator=used)
        cited = cited_indices(text, len(hits))
        citations = [self._citation(i, hits[i - 1]) for i in cited]
        return AgentReply(reply=text, escalate=False, citations=citations, generator=used)

    def _generate(self, message: str, hits: list[Hit]) -> tuple[str | None, str]:
        if not isinstance(self.generator, ExtractiveGenerator):
            try:
                text = self.generator.answer(message, hits)
                if text.strip() == "NO_ANSWER":
                    return None, self.generator.name
                if cited_indices(text, len(hits)):
                    return text, self.generator.name
                log.warning("LLM answer had no valid citation; using extractive fallback")
            except Exception as exc:  # network error, bad key, rate limit...
                log.warning("LLM call failed (%s); using extractive fallback", exc)
        return self.fallback.answer(message, hits), self.fallback.name

    @staticmethod
    def _citation(n: int, hit: Hit) -> Citation:
        c = hit.chunk
        return Citation(n=n, title=c.title, section=c.section, url=c.source, retrieved=c.retrieved)
