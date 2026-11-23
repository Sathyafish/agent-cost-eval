# agent-cost-eval

The same LangGraph support agent, logged two ways in Langfuse, running on a local Ollama model.

- **Agent A (coarse):** one generation for the whole run with summed tokens (for example "in 33,000 · out 410"). Tool calls appear as empty spans. You can't tell where the tokens went.
- **Agent B (granular):** an agent span, one generation per reasoning step with its own tokens, and a span per tool call with its full output. You can see that the step after `get_invoices` is the expensive one.

Both answer "Why was I charged twice this month?" with the same model (`llama3.1:8b`, temperature 0) and mocked tools. Only the logging differs.

## Run it

1. `ollama pull llama3.1:8b`
2. `docker compose up -d`, open http://localhost:3000, create a project and API keys
3. `cp .env.example .env` and fill in `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
4. `uv sync && uv run python -m agent_cost_eval.pricing`
5. `uv run python -m agent_cost_eval.compare`, then open the two printed trace URLs

Prices in `.env` are placeholders (Ollama is free). They exist so Langfuse shows dollar costs.
The compose file is for local use: replace its `# CHANGEME` secrets before exposing it.

## What to look for

- **Trace B:** the generation after `get_invoices` has by far the biggest input (about 12k tokens), and the `get_invoices` span holds the full 24-invoice output.
- **Trace A:** only the total tokens and cost. The `?` from the slide.
- A's total equals the sum of B's steps.

## Tests

```bash
uv run pytest                                   # unit tests, offline
uv run pytest -m integration -o addopts=''      # needs Ollama + Langfuse running
```

If the step after `get_invoices` shows few input tokens, the payload was truncated: raise `NUM_CTX`.
