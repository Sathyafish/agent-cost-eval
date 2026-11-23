# Agent cost eval: coarse vs granular tracing (design)

Date: 2026-11-22

## Purpose

Show, side by side in Langfuse, the difference between two ways of logging the same LLM agent run:

- **Agent A (coarse):** the whole run is one generation with summed tokens. You can't tell where the tokens went.
- **Agent B (granular):** the run is an agent span with one generation per reasoning step (own input/output tokens) and one span per tool call with its full output captured. You can see which step drives the cost.

The scenario mirrors the reference slides: a support agent answering "Why was I charged twice this month?" using `lookup_account`, `get_invoices` (large payload, about 10k tokens) and `search_help_center`.

## Decisions

- **Langfuse:** self-hosted via Docker Compose.
- **Stack:** Python with LangChain/LangGraph (ReAct agent) and `ChatOllama`.
- **Model:** `llama3.1:8b` on Ollama (host install), with `num_ctx` raised so it can read the large invoice payload. Model name is a config value.
- **Approach:** one shared agent definition, two observability modes. Only the logging differs, so any gap is purely a visibility gap.
- **Cost:** Ollama is free, so a configurable per-token price is registered in Langfuse for the model. Defaults are placeholders, labeled as such.

## Success criteria

- Unit tests pass offline.
- One real `compare.py` run produces two traces in Langfuse.
- In trace B, the `get_invoices`-driven step is visibly the largest input and its tool output is fully captured.
- In trace A, only the total tokens and cost are visible.
- A's total tokens equal the sum of B's steps (within a small tolerance; temperature 0).

## Layout

```
agent-cost-eval/
├── docker-compose.yml        # Langfuse (web, worker, postgres, clickhouse, redis, minio)
├── .env.example              # LANGFUSE_* keys/host, OLLAMA_MODEL, num_ctx, token prices
├── pyproject.toml
├── README.md                 # run guide
├── src/agent_cost_eval/
│   ├── config.py             # reads .env: model, num_ctx, prices, Langfuse host
│   ├── tools.py              # lookup_account, get_invoices, search_help_center (mock, deterministic)
│   ├── agent.py              # build_agent(): the shared LangGraph ReAct agent
│   ├── coarse.py             # Agent A: run with no handler, log ONE generation
│   ├── granular.py           # Agent B: run with Langfuse CallbackHandler
│   ├── pricing.py            # registers the model price in Langfuse via API
│   └── compare.py            # CLI: same question to A and B, prints trace URLs
└── tests/
```

## Components

- **config.py:** loads `.env` with defaults. Single source for model, `num_ctx`, prices, Langfuse host and keys.
- **tools.py:** deterministic mock data, no network. `get_invoices` returns 24 invoices over two years (at least about 9,000 tokens). `lookup_account` and `search_help_center` return small stable payloads.
- **agent.py:** the only place the agent is defined. Temperature 0. Both modes call it.
- **coarse.py:** opens one Langfuse generation for the run. Records summed input/output tokens (from model usage metadata) and the final answer. Tool names appear as child spans with no output and no tokens.
- **granular.py:** passes Langfuse's `CallbackHandler` to the run. Langfuse records the agent span, a generation per step with its own tokens, and a span per tool call with full output.
- **pricing.py:** one-time setup registering `llama3.1:8b` with input/output per-token prices from `.env` through the Langfuse model-definitions API.
- **compare.py:** preflight, runs A then B sequentially with the same question and settings, tags both (`run_id`, `variant=coarse|granular`), flushes Langfuse, prints both trace URLs and a console summary (total tokens per run; per-step breakdown for B).

## Data flow

1. `compare.py` loads config and checks Ollama and Langfuse are reachable. It fails fast with a clear message otherwise.
2. Agent A runs, then Agent B, sequentially.
3. Each run follows the ReAct loop: reason, `lookup_account`, reason, `get_invoices`, reason, `search_help_center`, respond.
4. Token usage comes from Ollama (`prompt_eval_count`, `eval_count`) via `ChatOllama` usage metadata. A sums it; B records it per step.
5. Langfuse is flushed and trace URLs plus the summary are printed.

## Error handling

- **Ollama down or model missing:** stop before any run; say which, and suggest `ollama pull llama3.1:8b`.
- **Langfuse unreachable or bad keys:** stop before any run.
- **Tool or model failure mid-run:** record the exception on the trace, mark the run failed, and still run the other agent.
- **Missing usage metadata:** warn and log zero rather than guessing.

## Testing (pytest)

- **Unit (no services):**
  - `get_invoices` returns 24 invoices and a payload of at least about 9,000 tokens; other tools are stable.
  - Config loads `.env` values and applies defaults.
  - Coarse wrapper's total equals the sum of fake per-step usage.
- **Integration (`@pytest.mark.integration`, skipped unless Ollama and Langfuse are up):**
  - After a granular run, the Langfuse API shows an agent span, at least 3 generations, 3 tool spans, and the full `get_invoices` output.
  - After a coarse run, it shows one generation and tool spans without output or per-step tokens.
  - Coarse total tokens match the granular sum within tolerance.

## Run guide (README)

1. `ollama pull llama3.1:8b`
2. `docker compose up -d`; create a Langfuse project and keys at `localhost:3000`
3. Copy `.env.example` to `.env` and fill in keys
4. `uv sync && python -m agent_cost_eval.pricing`
5. `python -m agent_cost_eval.compare`; open the two printed URLs and compare

## Out of scope

UI, multiple scenarios, evaluations or scoring, CI pipeline, non-Ollama providers.
