import logging

from langchain_core.messages import AIMessage

from agent_cost_eval.agent import StepUsage, usage_of
from agent_cost_eval.coarse import sum_usage


def test_sum_usage():
    steps = [
        StepUsage(1, 2100, 60),
        StepUsage(2, 2600, 70),
        StepUsage(3, 12500, 80),
        StepUsage(4, 15800, 200),
    ]
    assert sum_usage(steps) == (33000, 410)


def test_sum_usage_missing_metadata_warns(caplog):
    msg = AIMessage(content="no usage here")
    with caplog.at_level(logging.WARNING):
        i, o = usage_of(msg)
    assert (i, o) == (0, 0)
    assert "No usage metadata" in caplog.text
    assert sum_usage([StepUsage(1, i, o)]) == (0, 0)
