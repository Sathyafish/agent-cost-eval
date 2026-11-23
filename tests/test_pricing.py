import json

import responses

from agent_cost_eval.config import load_settings
from agent_cost_eval.pricing import register_model_price


@responses.activate
def test_registers_model_price():
    responses.add(responses.POST, "http://localhost:3000/api/public/models", json={"id": "m1"})
    s = load_settings({"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"})
    register_model_price(s)
    body = json.loads(responses.calls[0].request.body)
    assert body["modelName"] == "llama3.1:8b"
    assert body["unit"] == "TOKENS"
    assert body["inputPrice"] == s.input_price_per_token
    assert body["outputPrice"] == s.output_price_per_token
    assert body["matchPattern"] == r"(?i)^llama3\.1:8b$"
