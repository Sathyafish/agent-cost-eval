from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    ollama_model: str
    ollama_host: str
    num_ctx: int
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str
    input_price_per_token: float
    output_price_per_token: float


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    if env is None:
        load_dotenv()
        env = os.environ
    return Settings(
        ollama_model=env.get("OLLAMA_MODEL", "llama3.1:8b"),
        ollama_host=env.get("OLLAMA_HOST", "http://localhost:11434"),
        num_ctx=int(env.get("NUM_CTX", "16384")),
        langfuse_host=env.get("LANGFUSE_HOST", "http://localhost:3000"),
        langfuse_public_key=env.get("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=env.get("LANGFUSE_SECRET_KEY", ""),
        # Placeholder prices (USD per token), only so Langfuse shows dollar costs.
        input_price_per_token=float(env.get("INPUT_PRICE_PER_TOKEN", "0.000003")),
        output_price_per_token=float(env.get("OUTPUT_PRICE_PER_TOKEN", "0.000015")),
    )
