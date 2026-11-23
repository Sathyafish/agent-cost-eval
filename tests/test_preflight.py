import pytest
import responses

from agent_cost_eval.config import load_settings
from agent_cost_eval.preflight import PreflightError, preflight

ENV = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
LF = "http://localhost:3000/api/public/health"
OL = "http://localhost:11434/api/tags"


def _ok(rsps, models=("llama3.1:8b",)):
    rsps.add(responses.GET, LF, json={"status": "OK"})
    rsps.add(responses.GET, OL, json={"models": [{"name": m} for m in models]})


@responses.activate
def test_healthy_returns_none():
    _ok(responses)
    assert preflight(load_settings(ENV)) is None


def test_missing_keys():
    with pytest.raises(PreflightError, match="LANGFUSE_PUBLIC_KEY"):
        preflight(load_settings({}))


@responses.activate
def test_langfuse_unreachable():
    responses.add(responses.GET, OL, json={"models": []})
    # no LF mock registered -> connection error
    with pytest.raises(PreflightError, match="Langfuse"):
        preflight(load_settings(ENV))


@responses.activate
def test_ollama_unreachable():
    responses.add(responses.GET, LF, json={"status": "OK"})
    with pytest.raises(PreflightError, match="Ollama"):
        preflight(load_settings(ENV))


@responses.activate
def test_model_missing():
    _ok(responses, models=("other:1b",))
    with pytest.raises(PreflightError, match="ollama pull llama3.1:8b"):
        preflight(load_settings(ENV))
