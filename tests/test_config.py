from agent_cost_eval.config import Settings, load_settings


def test_defaults():
    s = load_settings({})
    assert isinstance(s, Settings)
    assert s.ollama_model == "llama3.1:8b"
    assert s.num_ctx == 16384
    assert s.langfuse_host == "http://localhost:3000"
    assert s.ollama_host == "http://localhost:11434"


def test_overrides():
    s = load_settings(
        {
            "OLLAMA_MODEL": "qwen2.5:7b",
            "NUM_CTX": "8192",
            "INPUT_PRICE_PER_TOKEN": "0.000002",
        }
    )
    assert s.ollama_model == "qwen2.5:7b"
    assert s.num_ctx == 8192 and isinstance(s.num_ctx, int)
    assert s.input_price_per_token == 0.000002
    assert isinstance(s.input_price_per_token, float)
