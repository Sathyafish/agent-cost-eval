from agent_cost_eval.agent import QUESTION, run_agent
from agent_cost_eval.config import load_settings
from tests.fakes import ScriptedModel, ai, slide_script


def test_run_collects_steps_and_tool_names():
    llm = ScriptedModel(script=slide_script())
    result = run_agent(load_settings({}), llm=llm)
    assert result.failed is False
    assert result.tool_names == ["lookup_account", "get_invoices", "search_help_center"]
    assert [(s.input_tokens, s.output_tokens) for s in result.steps] == [
        (2100, 60), (2600, 70), (12500, 80), (15800, 200)
    ]
    assert result.answer.startswith("You were charged twice")


def test_recursion_limit_marks_failed():
    llm = ScriptedModel(script=[ai("lookup_account", 10, 5)], loop=True)
    result = run_agent(load_settings({}), question=QUESTION, llm=llm)
    assert result.failed is True
    assert result.error
