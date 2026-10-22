import pytest

from app.agent import escalation

SENSITIVE = [
    ("I'm on academic probation", "academic_standing"),
    ("I got dismissed from the university", "academic_standing"),
    ("How do I appeal my grade?", "academic_standing"),
    ("I can't afford tuition", "financial_hardship"),
    ("I lost my job and I'm about to be evicted", "financial_hardship"),
    ("I was accused of cheating on the exam", "disciplinary"),
    ("This is a Title IX issue", "disciplinary"),
    ("I want to complain about my professor", "complaint"),
    ("My SEVIS record was terminated", "immigration_status"),
    ("I'm out of status", "immigration_status"),
    ("Let me talk to a human", "human_request"),
    ("I want to kill myself", "personal_safety"),
    ("I'm thinking about suicide", "personal_safety"),
    ("I don’t want to live anymore", "personal_safety"),  # curly apostrophe
]

ROUTINE = [
    "How do I drop a class?",
    "What is the refund if I withdraw?",
    "How do I pay tuition?",
    "Is there a payment plan?",
    "Can I work on campus?",
    "My portal account is suspended, how do I log in?",
    "What is the policy on dismissal of class early?",
]


@pytest.mark.parametrize("message,category", SENSITIVE)
def test_sensitive_messages_always_escalate(message, category):
    result = escalation.check(message)
    assert result is not None, message
    assert result.category == category


@pytest.mark.parametrize("message", ROUTINE)
def test_routine_questions_do_not_escalate(message):
    assert escalation.check(message) is None


def test_safety_is_urgent_and_gives_crisis_resources():
    result = escalation.check("I want to hurt myself")
    assert result.urgent
    assert "988" in result.message and "911" in result.message


def test_safety_wins_over_other_categories():
    # A message that matches several rules must be treated as the most serious one.
    result = escalation.check("I'm on probation and I want to kill myself")
    assert result.category == "personal_safety"
