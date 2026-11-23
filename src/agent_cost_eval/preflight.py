"""Fail fast, with an actionable message, before any agent run."""
from __future__ import annotations

import requests

from agent_cost_eval.config import Settings


class PreflightError(Exception):
    pass


def preflight(settings: Settings) -> None:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        raise PreflightError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are not set. Create a project at "
            f"{settings.langfuse_host} and put its keys in .env."
        )
    try:
        requests.get(f"{settings.langfuse_host}/api/public/health", timeout=5).raise_for_status()
    except requests.RequestException as exc:
        raise PreflightError(
            f"Langfuse is not reachable at {settings.langfuse_host} ({exc}). Run: docker compose up -d"
        ) from exc
    try:
        resp = requests.get(f"{settings.ollama_host}/api/tags", timeout=5)
        resp.raise_for_status()
        names = {m["name"] for m in resp.json().get("models", [])}
    except requests.RequestException as exc:
        raise PreflightError(
            f"Ollama is not reachable at {settings.ollama_host} ({exc}). Start it with: ollama serve"
        ) from exc
    if settings.ollama_model not in names:
        raise PreflightError(
            f"Model {settings.ollama_model} is not installed. Run: ollama pull {settings.ollama_model}"
        )
