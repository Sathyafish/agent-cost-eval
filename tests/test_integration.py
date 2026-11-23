"""Needs live Ollama + self-hosted Langfuse. Run: uv run pytest -m integration -o addopts=''"""
import time

import pytest
import requests
from langfuse import get_client

from agent_cost_eval.agent import QUESTION
from agent_cost_eval.config import load_settings
from agent_cost_eval.preflight import PreflightError, preflight
from agent_cost_eval.pricing import register_model_price
from agent_cost_eval.support import run_support_agent

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    try:
        preflight(s)
    except PreflightError as exc:
        pytest.skip(str(exc))
    return s


def _get(s, path, **params):
    r = requests.get(
        f"{s.langfuse_host}{path}",
        params=params,
        auth=(s.langfuse_public_key, s.langfuse_secret_key),
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _observations(s, trace_id, expect_min, timeout=30):
    deadline = time.time() + timeout
    obs = []
    while time.time() < deadline:
        obs = _get(s, "/api/public/observations", traceId=trace_id, limit=100)["data"]
        if len(obs) >= expect_min:
            return obs
        time.sleep(2)
    return obs


@pytest.fixture(scope="module")
def run(settings):
    register_model_price(settings)
    result, trace_id = run_support_agent(settings, QUESTION, "itest-user", "itest-session")
    get_client().flush()
    return settings, result, trace_id


def test_trace_has_steps_tools_and_full_invoice_output(run):
    s, result, trace_id = run
    assert not result.failed
    obs = _observations(s, trace_id, expect_min=7)
    assert len([o for o in obs if o["type"] == "GENERATION"]) >= 3
    assert {"lookup_account", "get_invoices", "search_help_center"} <= {o["name"] for o in obs}
    invoices_out = next(o for o in obs if o["name"] == "get_invoices")["output"]
    assert str(invoices_out).count("inv_") >= 24


def test_trace_is_tagged_with_user_and_session(run):
    s, _, trace_id = run
    trace = _get(s, f"/api/public/traces/{trace_id}")
    assert trace["userId"] == "itest-user"
    assert trace["sessionId"] == "itest-session"


def test_generations_have_nonzero_cost_once_priced(run):
    s, _, trace_id = run
    gens = [o for o in _observations(s, trace_id, expect_min=7) if o["type"] == "GENERATION"]
    assert gens and all((g.get("calculatedTotalCost") or 0) > 0 for g in gens)
