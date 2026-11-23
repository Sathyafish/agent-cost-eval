from agent_cost_eval import compare
from agent_cost_eval.agent import AgentResult, StepUsage


class _Client:
    def flush(self):
        pass


def test_one_failure_does_not_stop_the_other(monkeypatch, capsys):
    ok = AgentResult("answer", [StepUsage(1, 100, 10, [])], [])

    def boom(*a, **k):
        raise RuntimeError("tool exploded")

    monkeypatch.setattr(compare, "preflight", lambda s: None)
    monkeypatch.setattr(compare, "run_coarse", boom)
    monkeypatch.setattr(compare, "run_granular", lambda s, rid: (ok, "tid123"))
    monkeypatch.setattr(compare, "get_client", lambda: _Client())

    code = compare.main()
    out = capsys.readouterr().out
    assert code != 0
    assert "/trace/tid123" in out
    assert "tool exploded" in out
