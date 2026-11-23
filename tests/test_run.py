from agent_cost_eval import run
from agent_cost_eval.agent import AgentResult, StepUsage

OK = AgentResult("answer", [StepUsage(1, 100, 10, [])], [])


def _patch(monkeypatch, fn):
    monkeypatch.setattr(run, "preflight", lambda s: None)
    monkeypatch.setattr(run, "run_support_agent", fn)
    monkeypatch.setattr(run, "get_client", lambda: type("C", (), {"flush": lambda self: None})())


def test_five_questions_each_with_user_and_session(monkeypatch):
    calls = []

    def fake(settings, question, user_id, session_id, llm=None):
        calls.append((question, user_id, session_id))
        return OK, f"t{len(calls)}"

    _patch(monkeypatch, fake)
    assert run.main() == 0
    assert len(run.SAMPLE_QUESTIONS) == 5 and len(calls) == 5
    assert all(user and session for _, user, session in calls)
    assert len({session for *_, session in calls}) == 5
    assert len({user for _, user, _ in calls}) >= 2  # cost-per-user needs more than one user


def test_one_failure_does_not_stop_the_rest(monkeypatch, capsys):
    n = {"i": 0}

    def fake(settings, question, user_id, session_id, llm=None):
        n["i"] += 1
        if n["i"] == 2:
            raise RuntimeError("tool exploded")
        return OK, f"tid{n['i']}"

    _patch(monkeypatch, fake)
    code = run.main()
    out = capsys.readouterr().out
    assert code != 0
    assert "/trace/tid5" in out and "tool exploded" in out
