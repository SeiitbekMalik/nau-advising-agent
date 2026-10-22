from fastapi.testclient import TestClient

from app import main


def client() -> TestClient:
    main.get_store.cache_clear()
    main.get_agent.cache_clear()
    return TestClient(main.app)


def test_health():
    r = client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["knowledge_base_chunks"] > 30


def test_advising_message_contract():
    c = client()
    r = c.post("/advising-message", json={"channel": "web", "student_id": "api-1", "message": "How do I drop a class?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"reply", "escalate", "citations", "generator", "latency_ms"}
    assert body["escalate"] is False
    assert body["citations"][0]["url"].startswith("https://www.na.edu/")


def test_escalation_flow_through_api():
    c = client()
    r = c.post("/advising-message", json={"student_id": "api-2", "message": "I can't afford tuition anymore"})
    assert r.json()["escalate"] is True
    queue = c.get("/escalations").json()
    item = next(e for e in queue if e["student_id"] == "api-2")
    assert c.post(f"/escalations/{item['id']}/close").status_code == 200
    assert all(e["student_id"] != "api-2" for e in c.get("/escalations").json())


def test_history_endpoint():
    c = client()
    c.post("/advising-message", json={"student_id": "api-3", "message": "How do I reset my password?"})
    hist = c.get("/students/api-3/history").json()
    assert len(hist) == 2 and hist[0]["role"] == "student"


def test_input_validation():
    c = client()
    assert c.post("/advising-message", json={"student_id": "x", "message": ""}).status_code == 422
    assert c.post("/advising-message", json={"student_id": "x", "message": "hi", "channel": "fax"}).status_code == 422


def test_pages_are_served():
    c = client()
    assert "Advising Assistant" in c.get("/").text
    assert "Advisor queue" in c.get("/advisor").text
