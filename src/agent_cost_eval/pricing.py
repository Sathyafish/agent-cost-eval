"""One-time setup: register a per-token price for the Ollama model in Langfuse."""
from __future__ import annotations

import re

import requests

from agent_cost_eval.config import Settings, load_settings


def _already_registered(settings: Settings) -> bool:
    auth = (settings.langfuse_public_key, settings.langfuse_secret_key)
    page = 1
    while True:
        resp = requests.get(
            f"{settings.langfuse_host}/api/public/models",
            params={"page": page, "limit": 100},
            auth=auth,
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        if any(m.get("modelName") == settings.ollama_model for m in payload.get("data", [])):
            return True
        if page >= payload.get("meta", {}).get("totalPages", 1):
            return False
        page += 1


def register_model_price(settings: Settings) -> None:
    """Register the price once; safe to re-run."""
    if _already_registered(settings):
        return
    body = {
        "modelName": settings.ollama_model,
        "matchPattern": f"(?i)^{re.escape(settings.ollama_model)}$",
        "unit": "TOKENS",
        "inputPrice": settings.input_price_per_token,
        "outputPrice": settings.output_price_per_token,
    }
    resp = requests.post(
        f"{settings.langfuse_host}/api/public/models",
        json=body,
        auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
        timeout=15,
    )
    resp.raise_for_status()


def main() -> None:
    s = load_settings()
    register_model_price(s)
    print(
        f"Registered {s.ollama_model}: input ${s.input_price_per_token}/token, "
        f"output ${s.output_price_per_token}/token (placeholder prices)"
    )


if __name__ == "__main__":
    main()
