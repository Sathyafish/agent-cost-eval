from fastapi.testclient import TestClient

from agent_cost_eval import service
from agent_cost_eval.agent import AgentResult, StepUsage

OK = AgentResult("Refund in 5 days.", [StepUsage(1, 120, 12, [])], [])


def test_ask_returns_answer_and_trace_url(monkeypatch):
    seen = {}

    def fake(settings, question, user_id, session_id, llm=None):
        seen.update(question=question, user_id=user_id, session_id=session_id)
        return OK, "abc123"

    monkeypatch.setattr(service, "run_support_agent", fake)
    client = TestClient(service.app)
    r = client.post("/ask", json={"question": "Charged twice?", "user_id": "u1", "session_id": "s1"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Refund in 5 days."
    assert body["trace_url"].endswith("/trace/abc123")
    assert body["input_tokens"] == 120 and body["output_tokens"] == 12
    assert seen == {"question": "Charged twice?", "user_id": "u1", "session_id": "s1"}


def test_empty_question_rejected(monkeypatch):
    monkeypatch.setattr(service, "run_support_agent", lambda *a, **k: (OK, "x"))
    r = TestClient(service.app).post("/ask", json={"question": "  ", "user_id": "u", "session_id": "s"})
    assert r.status_code == 422


def test_failed_run_returns_502(monkeypatch):
    bad = AgentResult("", [], [], failed=True, error="RecursionError: limit")
    monkeypatch.setattr(service, "run_support_agent", lambda *a, **k: (bad, "t9"))
    r = TestClient(service.app).post("/ask", json={"question": "hi", "user_id": "u", "session_id": "s"})
    assert r.status_code == 502
    assert "RecursionError" in r.json()["detail"]


def test_health():
    assert TestClient(service.app).get("/health").json() == {"status": "ok"}
