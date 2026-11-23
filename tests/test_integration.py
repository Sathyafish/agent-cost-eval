"""Needs live Ollama + self-hosted Langfuse. Run: uv run pytest -m integration -o addopts=''"""
import json
import time

import pytest
import requests
from langfuse import get_client

from agent_cost_eval.coarse import run_coarse, sum_usage
from agent_cost_eval.config import load_settings
from agent_cost_eval.granular import run_granular
from agent_cost_eval.preflight import PreflightError, preflight

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    try:
        preflight(s)
    except PreflightError as exc:
        pytest.skip(str(exc))
    return s


def _observations(s, trace_id, expect_min, timeout=30):
    deadline = time.time() + timeout
    obs = []
    while time.time() < deadline:
        r = requests.get(
            f"{s.langfuse_host}/api/public/observations",
            params={"traceId": trace_id, "limit": 100},
            auth=(s.langfuse_public_key, s.langfuse_secret_key),
            timeout=15,
        )
        r.raise_for_status()
        obs = r.json()["data"]
        if len(obs) >= expect_min:
            return obs
        time.sleep(2)
    return obs


@pytest.fixture(scope="module")
def runs(settings):
    granular, g_id = run_granular(settings, "itest")
    coarse, c_id = run_coarse(settings, "itest")
    get_client().flush()
    return settings, granular, g_id, coarse, c_id


def test_granular_trace_shape(runs):
    s, granular, g_id, _, _ = runs
    obs = _observations(s, g_id, expect_min=7)
    gens = [o for o in obs if o["type"] == "GENERATION"]
    assert len(gens) >= 3
    names = {o["name"] for o in obs}
    assert {"lookup_account", "get_invoices", "search_help_center"} <= names
    invoices_out = next(o for o in obs if o["name"] == "get_invoices")["output"]
    assert str(invoices_out).count("inv_") >= 24


def test_coarse_trace_shape(runs):
    s, _, _, _, c_id = runs
    obs = _observations(s, c_id, expect_min=4)
    assert len([o for o in obs if o["type"] == "GENERATION"]) == 1
    tool_spans = [o for o in obs if o["name"].startswith("tool:")]
    assert len(tool_spans) >= 3
    assert all(not o.get("output") for o in tool_spans)


def test_totals_match(runs):
    _, granular, _, coarse, _ = runs
    g = sum(sum_usage(granular.steps))
    c = sum(sum_usage(coarse.steps))
    assert c == pytest.approx(g, rel=0.05)
