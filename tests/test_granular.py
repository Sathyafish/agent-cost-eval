import logging

from agent_cost_eval.agent import StepUsage
from agent_cost_eval.granular import check_payload_visible


def _steps(after_invoices_input: int):
    return [
        StepUsage(1, 2100, 60, ["lookup_account"]),
        StepUsage(2, 2600, 70, ["get_invoices"]),
        StepUsage(3, after_invoices_input, 80, ["search_help_center"]),
        StepUsage(4, 15800, 200, []),
    ]


def test_payload_visible_true():
    assert check_payload_visible(_steps(12500)) is True


def test_payload_visible_false_warns(caplog):
    with caplog.at_level(logging.WARNING):
        assert check_payload_visible(_steps(3000)) is False
    assert "num_ctx" in caplog.text
