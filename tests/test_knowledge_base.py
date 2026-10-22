from app.kb.loader import parse_markdown
from app.kb.retriever import BM25Retriever, tokenize


def test_every_chunk_has_an_official_source(chunks):
    assert len(chunks) > 30
    for c in chunks:
        assert c.source.startswith("https://www.na.edu/"), c.id
        assert c.retrieved, c.id
        assert len(c.text) > 40, c.id


def test_chunk_ids_are_unique(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_parse_markdown_front_matter_and_sections(tmp_path):
    f = tmp_path / "demo.md"
    f.write_text("---\ntitle: Demo Office\nsource: https://www.na.edu/demo/\nretrieved: 2026-10-19\n---\n\n"
                 "## Hours\nOpen 9 to 5.\n\n## Email\nWrite to demo@na.edu.\n", encoding="utf-8")
    chunks = parse_markdown(f)
    assert [c.section for c in chunks] == ["Hours", "Email"]
    assert chunks[0].title == "Demo Office"
    assert chunks[1].id == "demo#email"


def test_tokenizer_normalizes_student_words():
    assert "course" in tokenize("Dropping my classes")
    assert "f1" in tokenize("Can I work on an F-1 visa?")
    assert "i20" in tokenize("extend my I-20")
    # "full-time" and "full time" must match each other
    assert set(tokenize("full-time")) == set(tokenize("full time"))


def test_retriever_finds_the_right_section(chunks):
    r = BM25Retriever(chunks)
    assert r.search("how much is a transcript", 1)[0].chunk.id == "registrar#ordering-an-official-transcript"
    assert r.search("reset password", 1)[0].chunk.id == "it-services#resetting-your-password"
    assert r.search("on campus job F-1", 1)[0].chunk.id == "international-student-office#on-campus-work-for-f-1-students"


def test_retriever_is_not_confident_about_unknown_topics(chunks):
    hits = BM25Retriever(chunks).search("what time does the gym open", 1)
    assert not hits or not hits[0].confident
