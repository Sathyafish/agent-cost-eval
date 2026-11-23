import json

import responses

from agent_cost_eval.config import load_settings
from agent_cost_eval.pricing import register_model_price


@responses.activate
def test_registers_model_price():
    responses.add(responses.GET, "http://localhost:3000/api/public/models", json={"data": [], "meta": {"totalPages": 1}})
    responses.add(responses.POST, "http://localhost:3000/api/public/models", json={"id": "m1"})
    s = load_settings({"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"})
    register_model_price(s)
    body = json.loads(responses.calls[-1].request.body)
    assert body["modelName"] == "llama3.1:8b"
    assert body["unit"] == "TOKENS"
    assert body["inputPrice"] == s.input_price_per_token
    assert body["outputPrice"] == s.output_price_per_token
    assert body["matchPattern"] == r"(?i)^llama3\.1:8b$"


@responses.activate
def test_skips_when_model_already_registered():
    responses.add(
        responses.GET,
        "http://localhost:3000/api/public/models",
        json={"data": [{"modelName": "llama3.1:8b"}], "meta": {"totalPages": 1}},
    )
    s = load_settings({"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"})
    register_model_price(s)  # no POST registered: would raise ConnectionError if attempted
    assert [c.request.method for c in responses.calls] == ["GET"]
