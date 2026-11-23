# agent-cost-eval

A customer support agent built with LangChain/LangGraph, running on a local Ollama model (`llama3.1:8b`) and traced in a self-hosted Langfuse, so you can see exactly where tokens and cost go.

The agent answers billing questions using three mocked tools: `lookup_account`, `get_invoices` (a large payload, about 10k tokens) and `search_help_center`. Every question becomes one Langfuse trace with an agent span, one generation per reasoning step (own tokens and cost) and one span per tool call with its full output. Traces carry a `user_id` and `session_id`.

## Run locally

1. `ollama pull llama3.1:8b`
2. `docker compose up -d`, open http://localhost:3000, create a project and API keys
3. `cp .env.example .env` and fill in `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
4. `uv sync && uv run python -m agent_cost_eval.pricing` (registers the token price, safe to re-run)
5. `uv run python -m agent_cost_eval.run` (sends 5 sample questions from 3 users)
6. Or run the API: `uv run uvicorn agent_cost_eval.service:app --port 8000`, then `POST /ask` with `{"question", "user_id", "session_id"}`

To deploy on EC2, follow [deploy/EC2.md](deploy/EC2.md).

## Reading token cost in Langfuse

Ollama is free, so cost comes from the per-token price registered in step 4. The placeholder prices in `.env` only make dollars appear. Menu names may differ slightly between Langfuse versions.

1. **Per trace:** open **Tracing > Traces**. Each row is one customer question, with its token counts and total cost. Add the cost columns from the column picker if they're hidden.
2. **Per step:** open one trace. The tree shows each reasoning step as a generation with its own input tokens, output tokens and cost. The step right after `get_invoices` has by far the largest input (about 12k tokens), because the invoice payload is fed back to the model. That is the cost driver. Click the `get_invoices` span to see the full output that caused it.
3. **Per user and session:** in the traces table filter or group by User or Session. This is why each question is sent with a `user_id` and `session_id`.
4. **Over time and by model:** open **Dashboards**. The built-in dashboard shows cost by model, tokens, and cost over time. Use it to watch totals as you send more questions.
5. **If cost is $0 or blank:** the model price isn't registered, or the registered name doesn't match `OLLAMA_MODEL`. Re-run step 4. Under **Settings > Models** the model should list input and output prices.

## Tests

```bash
uv run pytest                                   # unit tests, offline
uv run pytest -m integration -o addopts=''      # needs Ollama + Langfuse running
```

If the step after `get_invoices` shows few input tokens, the payload was truncated: raise `NUM_CTX`.
