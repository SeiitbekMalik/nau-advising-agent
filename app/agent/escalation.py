"""Escalation layer: decide BEFORE answering whether a human must handle this.

Rule of the project: the agent never decides sensitive matters. Academic
standing, financial hardship, discipline, complaints, immigration-status
problems and anything involving personal safety go to a human, 100% of the
time. Rules run before retrieval and before the LLM, so no model output can
override them.

Rules are plain, auditable patterns (an advisor can read this file and see
exactly what triggers a handoff). Each category has its own human-facing
message and the office that should take the case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Escalation:
    category: str
    urgent: bool
    handoff_to: str
    message: str


# Order matters: the first matching category wins, so safety comes first.
_RULES: list[tuple[str, bool, str, str, list[str]]] = [
    (
        "personal_safety",
        True,
        "Campus security / emergency services",
        "I'm really sorry you're going through this. Please reach out for help right now: "
        "if you are in immediate danger, call 911. You can call or text 988 (Suicide & Crisis Lifeline) "
        "any time, day or night. I'm also flagging this conversation so a member of NAU staff can follow up with you.",
        [
            r"\bkill (myself|me)\b", r"\bsuicid", r"\bend (my|it) (life|all)\b", r"\bwant to die\b",
            r"\bself[- ]?harm", r"\bhurt(ing)? myself\b", r"\bdon'?t want to (live|be alive)\b",
            r"\bsomeone (is )?(threaten|hurt)", r"\bin danger\b", r"\bbeing (abused|harassed|stalked)\b",
        ],
    ),
    (
        "academic_standing",
        False,
        "Academic Advisor / Department Chair",
        "Questions about academic probation, suspension, dismissal or grade appeals depend on your personal record "
        "and must be reviewed by a person, so I won't make a judgment here. I've flagged this conversation for an "
        "academic advisor. In the meantime you can contact your Academic Advisor or Department Chair directly.",
        [
            r"\bprobation\b", r"\bacademic (warning|suspension|dismissal)\b",
            r"\b(suspended|dismissed) from (school|nau|the university|university|my program|classes)\b",
            r"\bkicked out\b", r"\bexpell", r"\bgrade (appeal|dispute|change)\b", r"\bappeal (my|a) grade\b",
            r"\bfail(ed|ing)? (all|most|every)", r"\bgpa (is )?(too low|below)\b",
        ],
    ),
    (
        "financial_hardship",
        False,
        "Financial Aid Office / Bursar's Office",
        "I'm sorry you're dealing with financial stress. Decisions about hardship, exceptions or special aid are made "
        "by people, not by me, so I've flagged this for staff to follow up. You can also contact the Financial Aid "
        "Office (financialaid@na.edu, (832) 464-8692) or the Bursar's Office about a tuition payment plan "
        "(sa@na.edu, 832-230-5155).",
        [
            r"\bcan'?t (afford|pay)\b", r"\bcannot (afford|pay)\b", r"\bunable to (afford|pay)\b",
            r"\bhardship\b", r"\bbroke\b", r"\blost my (job|income|housing)\b", r"\bevict", r"\bhomeless",
            r"\bno (money|food|place to (live|stay))\b", r"\bfood insecur", r"\bemergency (fund|aid|loan)\b",
        ],
    ),
    (
        "disciplinary",
        False,
        "Dean of Students",
        "Matters involving academic integrity, conduct or Title IX must be handled by the appropriate NAU staff, "
        "so I won't advise on the outcome. I've flagged this conversation for a human to follow up.",
        [
            r"\bdisciplin", r"\bmisconduct\b", r"\b(accused|caught) (of )?(cheating|plagiari)",
            r"\bplagiari[sz]", r"\bcheating\b", r"\bhonor code\b", r"\bacademic (dishonesty|integrity)\b",
            r"\btitle ix\b", r"\bharass", r"\bassault",
        ],
    ),
    (
        "complaint",
        False,
        "Student Services",
        "I'm sorry about your experience. Complaints should be reviewed by a person, so I've flagged this "
        "conversation for staff follow-up. If you share what happened, a staff member will get back to you.",
        [r"\bcomplain", r"\bdiscriminat", r"\bunfair(ly)?\b", r"\bfile a grievance\b", r"\bgrievance\b"],
    ),
    (
        "immigration_status",
        False,
        "International Student Office (iso@na.edu)",
        "Problems with your immigration status need to be reviewed by a Designated School Official, not by me. "
        "I've flagged this for the International Student Office. Please also contact them directly at iso@na.edu "
        "or (832) 230-5555, press 2.",
        [
            r"\bout of status\b", r"\bsevis (terminat|record)", r"\bterminated (in )?sevis\b", r"\bvisa (revoked|denied|cancel)",
            r"\bdeport", r"\b(lost|losing) (my )?(f-?1|status)\b", r"\bunauthori[sz]ed (work|employment)\b",
        ],
    ),
    (
        "human_request",
        False,
        "Academic Advisor",
        "Of course. I've flagged this conversation so a human advisor can contact you.",
        [
            r"\b(talk|speak|chat) (to|with) (a |an )?(human|person|advisor|adviser|someone|staff|real)",
            r"\breal (person|human)\b", r"\bhuman advisor\b", r"\bnot a (bot|robot)\b",
        ],
    ),
]

_COMPILED = [
    (name, urgent, office, msg, [re.compile(p, re.I) for p in patterns])
    for name, urgent, office, msg, patterns in _RULES
]


def check(message: str) -> Escalation | None:
    """Return an Escalation if the message must go to a human, else None."""
    text = message.replace("’", "'")
    for name, urgent, office, msg, patterns in _COMPILED:
        if any(p.search(text) for p in patterns):
            return Escalation(category=name, urgent=urgent, handoff_to=office, message=msg)
    return None


CATEGORIES = [r[0] for r in _RULES]
