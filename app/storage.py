"""Conversation log and escalation queue (SQLite).

- One history per student: every message (student and agent) is stored with
  its channel, citations and escalation flag.
- Escalated conversations land in an `escalations` queue that an advisor can
  review and close.
- PII safety: anything that looks like an SSN, a card number or a phone/ID
  number is masked BEFORE it is written, so the log never stores it.

SQLite keeps the prototype dependency-free; the schema maps 1:1 to PostgreSQL.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT NOT NULL,
    channel     TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('student', 'agent')),
    text        TEXT NOT NULL,
    citations   TEXT NOT NULL DEFAULT '[]',
    escalated   INTEGER NOT NULL DEFAULT 0,
    category    TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_student ON messages(student_id, id);

CREATE TABLE IF NOT EXISTS escalations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT NOT NULL,
    channel     TEXT NOT NULL,
    category    TEXT NOT NULL,
    urgent      INTEGER NOT NULL DEFAULT 0,
    handoff_to  TEXT NOT NULL,
    message     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    created_at  TEXT NOT NULL
);
"""

_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[NUMBER REDACTED]"),
    (re.compile(r"\b\d{9}\b"), "[NUMBER REDACTED]"),
    (re.compile(r"(?<!\w)\+?1?[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\w)"), "[PHONE REDACTED]"),
]


def redact(text: str) -> str:
    """Mask SSNs, card/account-like numbers and phone numbers typed by a student."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, db_path: Path | str):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def add_message(self, student_id: str, channel: str, role: str, text: str,
                    citations: list[dict] | None = None, escalated: bool = False, category: str | None = None) -> None:
        # Student text is redacted; agent text comes from official sources and is stored as is.
        stored = redact(text) if role == "student" else text
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (student_id, channel, role, text, citations, escalated, category, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (student_id, channel, role, stored, json.dumps(citations or []), int(escalated), category, _now()),
            )
            self._conn.commit()

    def add_escalation(self, student_id: str, channel: str, category: str, urgent: bool,
                       handoff_to: str, message: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO escalations (student_id, channel, category, urgent, handoff_to, message, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (student_id, channel, category, int(urgent), handoff_to, redact(message), _now()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def history(self, student_id: str, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE student_id = ? ORDER BY id DESC LIMIT ?", (student_id, limit)
            ).fetchall()
        out = [dict(r) | {"citations": json.loads(r["citations"]), "escalated": bool(r["escalated"])} for r in rows]
        return list(reversed(out))

    def escalations(self, status: str | None = "open") -> list[dict]:
        sql, args = "SELECT * FROM escalations", ()
        if status:
            sql, args = sql + " WHERE status = ?", (status,)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY urgent DESC, id DESC", args).fetchall()
        return [dict(r) | {"urgent": bool(r["urgent"])} for r in rows]

    def close_escalation(self, escalation_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE escalations SET status = 'closed' WHERE id = ?", (escalation_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def stats(self) -> dict:
        with self._lock:
            q = lambda sql: self._conn.execute(sql).fetchone()[0]  # noqa: E731
            return {
                "students": q("SELECT COUNT(DISTINCT student_id) FROM messages"),
                "questions": q("SELECT COUNT(*) FROM messages WHERE role = 'student'"),
                "escalations_open": q("SELECT COUNT(*) FROM escalations WHERE status = 'open'"),
                "escalations_total": q("SELECT COUNT(*) FROM escalations"),
            }
