from app.kb import ingest
from app.kb.loader import parse_markdown

HTML = """<html><head><title>Library | NAU</title></head><body>
<nav>Home | About</nav>
<main><h1>NAU Library</h1>
<p>The library supports students and faculty.</p>
<h2>Hours</h2><p>Monday to Friday, 8 am to 9 pm.</p><ul><li>Saturday 10 am to 4 pm</li></ul>
<h2>Borrowing</h2><p>Students may borrow up to 5 books for 3 weeks.</p>
</main><footer>Copyright</footer><script>var x=1;</script></body></html>"""


def test_html_to_sections_keeps_main_text_only():
    title, sections = ingest.html_to_sections(HTML)
    assert title == "NAU Library"
    headings = [h for h, _ in sections]
    assert headings == ["Overview", "Hours", "Borrowing"]
    text = " ".join(t for _, t in sections)
    assert "Saturday 10 am" in text
    assert "Copyright" not in text and "var x" not in text and "Home | About" not in text


def test_long_sections_are_split():
    parts = ingest._split_long("Policy", "word " * 500)
    assert len(parts) == 3 and parts[1][0] == "Policy (part 2)"


def test_written_markdown_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "OUT_DIR", tmp_path)
    path = ingest.write_markdown("https://www.na.edu/library/", "NAU Library",
                                 [("Hours", "Monday to Friday, 8 am to 9 pm.")])
    chunks = parse_markdown(path)
    assert chunks[0].source == "https://www.na.edu/library/"
    assert chunks[0].section == "Hours"


def test_private_systems_are_never_ingested():
    assert ingest.ingest("https://portal.na.edu/ics", client=None) is None
