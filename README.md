# NAU Student Advising & Help-Desk Agent

An AI assistant that answers current North American University students' routine questions
(registration, add/drop, transcripts, tuition, financial aid, F-1 rules, IT, tutoring) **from NAU's
official pages, with citations**, and hands sensitive cases to a **human advisor**.

Capstone project, Fall 2026 · Student: Seiitbek Sharshenbaev · Advisor: Dr. Azamat Zhamanov

## What it does

```
Student question
      │
      ▼
1. Escalation rules ──► sensitive? (probation, hardship, discipline, complaint,
      │                   immigration status, safety, "talk to a human")
      │                   └─► human advisor queue, never decided by the AI
      ▼
2. Retrieval (BM25) over the NAU knowledge base
      ▼
3. Confidence gate ──► not covered by official info? ──► "I don't know" + human
      ▼
4. Answer generation (LLM or offline extractive mode), ONLY from retrieved sources
      ▼
5. Citation check ──► answer without a valid [n] citation is rejected
      ▼
6. Conversation log (PII redacted) + escalation queue
```

- **Grounded answers with sources.** Every answer cites the official na.edu page it came from.
- **Human escalation.** Rule-based, runs before the AI, auditable in `app/agent/escalation.py`.
- **No guessing.** Weak matches get "I couldn't find this" and a human handoff.
- **AI disclosure.** The assistant says it is automated at the start of every conversation.
- **Privacy.** No real student records. SSNs, card numbers and phone numbers typed by a student
  are masked before anything is stored. Ingestion refuses the portal, Canvas and other private systems.
- **Works offline.** Without an API key it answers by quoting the most relevant official sentences.
  With a key it uses an LLM behind a thin adapter (Anthropic or OpenAI), falling back to offline
  mode if the call fails or the answer is not cited.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Open **http://localhost:8000** (student chat) and **http://localhost:8000/advisor** (advisor queue).
API docs: http://localhost:8000/docs

Optional: copy `.env.example` to `.env` and add an `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

## API (shared NAU agent contract)

`POST /advising-message`

```json
{ "channel": "web", "student_id": "demo-student-001", "message": "How do I drop a class?" }
```

Response: `reply`, `escalate` (bool), `category`, `handoff_to`, `urgent`, `citations[]`
(`n`, `title`, `section`, `url`, `retrieved`), `generator`, `latency_ms`.

Other endpoints: `GET /students/{id}/history`, `GET /escalations`,
`POST /escalations/{id}/close`, `GET /health`.

## Knowledge base

- `data/kb/*.md`: curated summaries of official NAU pages (Registrar, Bursar, Financial Aid,
  IT Services, International Student Office, Student Success), collected 2026-10-19.
  Each file records its source URL; each `##` section becomes one citable chunk.
- `python -m app.kb.ingest` downloads every public page listed in `data/sources.txt`
  (HTML and PDF, including the academic catalog) into `data/kb/web/`.

## Tests and evaluation

```bash
pytest                                 # 50 unit/API tests
python -m eval.run_eval                # main question set (43 questions)
python -m eval.run_eval --set holdout  # held-out informal questions (16)
```

Results (offline extractive mode, 2026-10-19):

| Metric | Main set (43) | Held-out set (16) |
|---|---|---|
| Answer cites the correct official source | 100% | 45.5% |
| Sensitive cases escalated correctly | 100% | 100% |
| Unknown questions answered "I don't know" (no guessing) | 100% | 100% |
| Median latency | < 1 ms | < 1 ms |

The main set was written alongside the knowledge base, so it overstates real performance. The
held-out set uses informal phrasing ("how do i get out of my bio class") and shows the main
limitation: keyword search misses wording that differs from the official pages. When it misses,
it usually hands off to a human rather than giving a wrong answer (4 of 6 misses).

## Project structure

```
app/
  main.py               FastAPI app and API contract
  config.py             settings (.env)
  storage.py            conversation log, escalation queue, PII redaction (SQLite)
  agent/core.py         the pipeline: escalate -> retrieve -> gate -> generate -> cite -> log
  agent/escalation.py   human-escalation rules
  agent/llm.py          LLM adapter (Anthropic / OpenAI / offline extractive)
  kb/loader.py          markdown -> citable chunks
  kb/retriever.py       BM25 retrieval with student-vocabulary synonyms
  kb/ingest.py          download official pages / PDFs into the knowledge base
  static/               student chat page and advisor page
data/kb/                knowledge base
data/sources.txt        official URLs to ingest
eval/                   evaluation question sets and runner
tests/                  pytest suite
```

## Roadmap

- Ingest the full academic catalog, academic calendar and student handbook.
- Semantic retrieval with embeddings (PostgreSQL + pgvector), combined with BM25 (hybrid search),
  to handle informal phrasing.
- Run evaluation with an LLM generator; add volunteer student test questions.
- Usability testing with volunteer students; final evaluation report.

## License

Apache-2.0
