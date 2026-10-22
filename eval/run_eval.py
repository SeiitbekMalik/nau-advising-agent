"""Evaluate the agent against eval/questions.yaml and print a report.

Metrics (the success criteria from the project proposal):
  - Answer accuracy:      answerable questions whose reply cites the expected source
  - Retrieval hit@k:      expected source is among the top-k retrieved chunks
  - Escalation recall:    sensitive questions routed to a human with the right category (target 100%)
  - False escalations:    answerable questions wrongly handed off
  - Out-of-scope safety:  unanswerable questions answered with "I don't know" instead of a guess
  - Latency:              median / max response time

Usage:  python -m eval.run_eval [--set holdout] [--json results.json]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import yaml

from app.agent.core import AdvisingAgent
from app.agent.llm import build_generator
from app.config import ROOT, settings
from app.kb.loader import load_kb
from app.kb.retriever import BM25Retriever
from app.storage import Store


def build_agent() -> AdvisingAgent:
    generator = build_generator(settings.resolved_provider(), settings.anthropic_api_key,
                                settings.openai_api_key, settings.llm_model)
    return AdvisingAgent(BM25Retriever(load_kb(settings.kb_dir)), generator, Store(":memory:"),
                         top_k=settings.top_k, min_coverage=settings.min_coverage)


def evaluate(agent: AdvisingAgent, cases: dict) -> dict:
    rows, latencies = [], []
    chunk_ids = {c.id: c for c in agent.retriever.chunks}

    def cited_ids(reply) -> set[str]:
        ids = set()
        for c in reply.citations:
            for cid, chunk in chunk_ids.items():
                if chunk.title == c.title and chunk.section == c.section:
                    ids.add(cid)
        return ids

    for i, case in enumerate(cases.get("answerable", [])):
        r = agent.handle(f"eval-a{i}", case["q"])
        latencies.append(r.latency_ms)
        retrieved = [h.chunk.id for h in agent.retriever.search(case["q"], k=settings.top_k)]
        expected = case["expect_source"]
        expected = set(expected if isinstance(expected, list) else [expected])
        rows.append({"type": "answerable", "q": case["q"], "expected": sorted(expected),
                     "cited": sorted(cited_ids(r)), "escalated": r.escalate,
                     "correct": bool(expected & cited_ids(r)) and not r.escalate,
                     "hit_at_k": bool(expected & set(retrieved))})

    for i, case in enumerate(cases.get("escalation", [])):
        r = agent.handle(f"eval-e{i}", case["q"])
        latencies.append(r.latency_ms)
        rows.append({"type": "escalation", "q": case["q"], "expected": case["expect_escalate"],
                     "got": r.category, "correct": r.escalate and r.category == case["expect_escalate"]})

    for i, case in enumerate(cases.get("out_of_scope", [])):
        r = agent.handle(f"eval-o{i}", case["q"])
        latencies.append(r.latency_ms)
        rows.append({"type": "out_of_scope", "q": case["q"], "expected": "no_grounded_answer",
                     "got": r.category, "correct": r.category == "no_grounded_answer"})

    def rate(kind: str, key: str = "correct") -> float:
        sel = [r for r in rows if r["type"] == kind]
        return round(100 * sum(r[key] for r in sel) / len(sel), 1) if sel else 0.0

    answerable = [r for r in rows if r["type"] == "answerable"]
    return {
        "generator": agent.generator.name,
        "questions": len(rows),
        "answer_accuracy_pct": rate("answerable"),
        "retrieval_hit_at_k_pct": rate("answerable", "hit_at_k"),
        "false_escalations": sum(r["escalated"] for r in answerable),
        "escalation_recall_pct": rate("escalation"),
        "out_of_scope_safe_pct": rate("out_of_scope"),
        "latency_ms_median": statistics.median(latencies) if latencies else 0,
        "latency_ms_max": max(latencies) if latencies else 0,
        "rows": rows,
    }


def main(argv: list[str]) -> int:
    name = argv[argv.index("--set") + 1] if "--set" in argv else "main"
    file = "questions.yaml" if name == "main" else f"questions_{name}.yaml"
    cases = yaml.safe_load((ROOT / "eval" / file).read_text(encoding="utf-8"))
    print(f"Test set: {file}")
    report = evaluate(build_agent(), cases)
    print(f"Generator: {report['generator']}   Questions: {report['questions']}")
    print(f"  Answer accuracy (cites the right official source): {report['answer_accuracy_pct']}%")
    print(f"  Retrieval hit@{settings.top_k}:                               {report['retrieval_hit_at_k_pct']}%")
    print(f"  Sensitive cases escalated correctly:               {report['escalation_recall_pct']}%  (target 100%)")
    print(f"  Out-of-scope answered safely ('I don't know'):     {report['out_of_scope_safe_pct']}%")
    print(f"  False escalations on answerable questions:         {report['false_escalations']}")
    print(f"  Latency: median {report['latency_ms_median']} ms, max {report['latency_ms_max']} ms")
    misses = [r for r in report["rows"] if not r["correct"]]
    if misses:
        print("\nMisses:")
        for r in misses:
            print(f"  [{r['type']}] {r['q']}\n      expected {r['expected']}, got {r.get('cited', r.get('got'))}")
    if "--json" in argv:
        out = Path(argv[argv.index("--json") + 1])
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
