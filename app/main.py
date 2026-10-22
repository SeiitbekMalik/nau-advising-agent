"""FastAPI application.

Endpoints
  GET  /                          student chat page
  GET  /advisor                   advisor page: escalation queue + conversation history
  POST /advising-message          main endpoint (shared NAU agent API contract)
  GET  /students/{id}/history     one student's conversation log
  GET  /escalations               open escalations (?status=all for every one)
  POST /escalations/{id}/close    mark an escalation as handled
  GET  /health                    status: knowledge base size, generator in use, counts

Run:  uvicorn app.main:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import __version__
from app.agent.core import AdvisingAgent
from app.agent.llm import build_generator
from app.config import settings
from app.kb.loader import load_kb
from app.kb.retriever import BM25Retriever
from app.storage import Store

STATIC = Path(__file__).parent / "static"


# ---- API contract -------------------------------------------------------------------------
class MessageIn(BaseModel):
    channel: Literal["web", "email", "sms", "voice"] = "web"
    student_id: str = Field(min_length=1, max_length=64, description="Synthetic/test student ID")
    message: str = Field(min_length=1, max_length=2000)


class CitationOut(BaseModel):
    n: int
    title: str
    section: str
    url: str
    retrieved: str


class MessageOut(BaseModel):
    reply: str
    escalate: bool
    category: str | None = None
    handoff_to: str | None = None
    urgent: bool = False
    citations: list[CitationOut] = []
    generator: str
    latency_ms: int


# ---- wiring -------------------------------------------------------------------------------
@lru_cache
def get_store() -> Store:
    return Store(settings.db_path)


@lru_cache
def get_agent() -> AdvisingAgent:
    chunks = load_kb(settings.kb_dir)
    generator = build_generator(settings.resolved_provider(), settings.anthropic_api_key,
                                settings.openai_api_key, settings.llm_model)
    return AdvisingAgent(BM25Retriever(chunks), generator, get_store(),
                         top_k=settings.top_k, min_coverage=settings.min_coverage)


app = FastAPI(title="NAU Student Advising & Help-Desk Agent", version=__version__)


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/advisor", include_in_schema=False)
def advisor_page() -> FileResponse:
    return FileResponse(STATIC / "advisor.html")


@app.post("/advising-message", response_model=MessageOut)
def advising_message(body: MessageIn) -> MessageOut:
    r = get_agent().handle(body.student_id.strip(), body.message.strip(), body.channel)
    return MessageOut(reply=r.reply, escalate=r.escalate, category=r.category, handoff_to=r.handoff_to,
                      urgent=r.urgent, citations=[CitationOut(**c.__dict__) for c in r.citations],
                      generator=r.generator, latency_ms=r.latency_ms)


@app.get("/students/{student_id}/history")
def history(student_id: str) -> list[dict]:
    return get_store().history(student_id)


@app.get("/escalations")
def escalations(status: str = "open") -> list[dict]:
    return get_store().escalations(None if status == "all" else status)


@app.post("/escalations/{escalation_id}/close")
def close_escalation(escalation_id: int) -> dict:
    if not get_store().close_escalation(escalation_id):
        raise HTTPException(404, "Escalation not found")
    return {"closed": escalation_id}


@app.get("/health")
def health() -> dict:
    agent = get_agent()
    return {
        "status": "ok",
        "version": __version__,
        "generator": agent.generator.name,
        "knowledge_base_chunks": len(agent.retriever.chunks),
        "knowledge_base_sources": sorted({c.source for c in agent.retriever.chunks}),
        **get_store().stats(),
    }
