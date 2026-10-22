from app.agent.core import DISCLOSURE, AdvisingAgent
from app.agent.llm import ExtractiveGenerator, cited_indices
from app.kb.retriever import BM25Retriever
from app.storage import redact


def test_answer_is_grounded_and_cited(agent):
    r = agent.handle("s1", "How much does an official transcript cost?")
    assert not r.escalate
    assert "$10" in r.reply
    assert r.citations and r.citations[0].url == "https://www.na.edu/students/registrar/"
    assert "[1]" in r.reply


def test_ai_disclosure_on_first_message_only(agent):
    first = agent.handle("s2", "How do I reset my password?")
    second = agent.handle("s2", "How do I contact the help desk?")
    assert first.reply.startswith(DISCLOSURE)
    assert not second.reply.startswith(DISCLOSURE)


def test_unknown_question_is_not_guessed(agent):
    r = agent.handle("s3", "What time does the gym open?")
    assert r.escalate and r.category == "no_grounded_answer"
    assert not r.citations


def test_sensitive_question_escalates_and_is_queued(agent, store):
    r = agent.handle("s4", "I'm on academic probation, what now?")
    assert r.escalate and r.category == "academic_standing"
    queue = store.escalations()
    assert len(queue) == 1 and queue[0]["student_id"] == "s4"


def test_conversation_history_is_logged(agent, store):
    agent.handle("s5", "How do I drop a class?")
    agent.handle("s5", "How do I pay tuition?")
    hist = store.history("s5")
    assert [m["role"] for m in hist] == ["student", "agent", "student", "agent"]
    assert hist[1]["citations"]


def test_pii_is_redacted_before_logging(agent, store):
    agent.handle("s6", "My SSN is 123-45-6789 and card 4111 1111 1111 1111, how do I pay tuition?")
    stored = store.history("s6")[0]["text"]
    assert "123-45-6789" not in stored and "4111" not in stored
    assert "[SSN REDACTED]" in stored


def test_redact_phone_numbers():
    assert "832" not in redact("call me at (832) 555-1234")


class BrokenLLM:
    name = "broken"

    def answer(self, question, hits):
        raise RuntimeError("API down")


class UncitedLLM:
    name = "uncited"

    def answer(self, question, hits):
        return "You can definitely get a full refund any time."  # no citation -> must be rejected


def test_llm_failure_falls_back_to_grounded_answer(chunks, store):
    agent = AdvisingAgent(BM25Retriever(chunks), BrokenLLM(), store)
    r = agent.handle("s7", "How much does a transcript cost?")
    assert r.generator == "extractive" and "$10" in r.reply


def test_uncited_llm_answer_is_rejected(chunks, store):
    agent = AdvisingAgent(BM25Retriever(chunks), UncitedLLM(), store)
    r = agent.handle("s8", "What is the refund if I withdraw?")
    assert "definitely" not in r.reply
    assert r.citations


def test_cited_indices_ignores_invalid_numbers():
    assert cited_indices("A [2] B [1] C [9] D [2]", 3) == [2, 1]


def test_extractive_generator_cites_every_source_it_uses(chunks):
    hits = BM25Retriever(chunks).search("How do I pay tuition?", 4)
    text = ExtractiveGenerator().answer("How do I pay tuition?", hits)
    assert cited_indices(text, len(hits))
