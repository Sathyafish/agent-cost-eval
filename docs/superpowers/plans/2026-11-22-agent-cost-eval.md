# Agent cost eval implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two identical LangGraph support agents that differ only in Langfuse logging (coarse vs granular), plus a CLI that runs both and prints their trace URLs.

**Architecture:** One shared ReAct agent (`agent.py`) runs against `llama3.1:8b` on Ollama with mocked tools. Agent B passes Langfuse's `CallbackHandler` for per-step spans. Agent A runs with no handler and a wrapper logs one summed generation. A pricing script registers a per-token price so Langfuse shows dollar costs.

**Tech Stack:** Python 3.11+, uv, langchain-ollama, langgraph, langfuse (v3 SDK), pytest, Docker Compose (self-hosted Langfuse), Ollama on the host.

**Spec:** `docs/superpowers/specs/2026-11-22-agent-cost-eval-design.md`

## Global Constraints

- Model: `llama3.1:8b`, temperature 0, `num_ctx` raised (default 16384), model name and `num_ctx` are config values.
- Langfuse is self-hosted via Docker Compose at `http://localhost:3000`.
- Question for both agents: "Why was I charged twice this month?"
- `get_invoices` returns 24 invoices over two years, payload at least about 9,000 tokens, deterministic, no network.
- Agents run sequentially (A then B), same question and settings, tagged `run_id` and `variant=coarse|granular`.
- Default token prices are placeholders and labeled as such.
- Out of scope: UI, multiple scenarios, evals/scoring, CI, non-Ollama providers.

## Review Focus

1. Ollama returns no usage metadata: warn and log zero, never guess (Task 4).
2. `num_ctx` too small silently truncates the invoice payload: the step after `get_invoices` must show input tokens above 8,000, else warn (Task 5).
3. Tool raises mid-run: exception recorded, run marked failed, other agent still runs (Task 6).
4. Model loops or never answers: `recursion_limit` stops the run and it is marked failed, no hang (Task 2).
5. Missing Langfuse keys or Ollama model: fail fast before any run with an actionable message (Task 6).

---

### Task 1: Project scaffold and config

**Files:**
- Create: `pyproject.toml`, `.env.example`, `.gitignore`, `src/agent_cost_eval/__init__.py`, `src/agent_cost_eval/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` (frozen dataclass) with fields `ollama_model: str`, `ollama_host: str`, `num_ctx: int`, `langfuse_host: str`, `langfuse_public_key: str`, `langfuse_secret_key: str`, `input_price_per_token: float`, `output_price_per_token: float`; and `load_settings(env: Mapping[str, str] | None = None) -> Settings`.

- [ ] **Step 1: Write failing tests** in `tests/test_config.py`: `test_defaults` asserts `load_settings({})` gives `ollama_model == "llama3.1:8b"`, `num_ctx == 16384`, `langfuse_host == "http://localhost:3000"`, `ollama_host == "http://localhost:11434"`; `test_overrides` asserts `OLLAMA_MODEL`, `NUM_CTX`, `INPUT_PRICE_PER_TOKEN` env values are read and typed (`int`, `float`).
- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Create scaffold.** `pyproject.toml` (src layout, deps above, `pytest` dev dep, marker `integration`), `.gitignore` (`.env`, `.venv`, `__pycache__`), `.env.example` listing every `Settings` field as env var (`LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `OLLAMA_MODEL`, `OLLAMA_HOST`, `NUM_CTX`, `INPUT_PRICE_PER_TOKEN`, `OUTPUT_PRICE_PER_TOKEN`) with a comment that prices are placeholders. Implement `load_settings` in `config.py`; when `env is None` read `os.environ` after `dotenv.load_dotenv()`. Keys default to `""` (validated in Task 6, not here).
- [ ] **Step 4: Run** `uv sync && uv run pytest tests/test_config.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: project scaffold and config"`

### Task 2: Mock tools

**Files:**
- Create: `src/agent_cost_eval/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: LangChain tools `lookup_account(email: str) -> dict`, `get_invoices(account_id: str) -> list[dict]`, `search_help_center(query: str) -> list[dict]`, and `TOOLS: list` containing all three.

- [ ] **Step 1: Write failing tests:** `test_get_invoices_count` asserts `len(get_invoices.invoke({"account_id": "acct_1"})) == 24`; `test_get_invoices_is_large` asserts `len(json.dumps(result)) >= 36000` (about 9,000 tokens at 4 chars/token); `test_deterministic` asserts two calls are equal; `test_double_charge_present` asserts two invoices share a month with `status == "paid"` (so the agent has something to find); `test_other_tools_small` asserts `lookup_account` and `search_help_center` outputs are under 2,000 chars.
- [ ] **Step 2: Run** `uv run pytest tests/test_tools.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement tools** with `@tool`. Invoices have `id`, `date`, `amount`, `status`, and a `lines` list of line items padded with deterministic descriptive text to reach the size floor. Include one duplicated monthly charge in the most recent month.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: mock support tools"` (after `git add`).

### Task 3: Shared agent

**Files:**
- Create: `src/agent_cost_eval/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `Settings`, `TOOLS`.
- Produces:
  - `QUESTION = "Why was I charged twice this month?"`
  - `@dataclass StepUsage: step: int; input_tokens: int; output_tokens: int; tool_calls: list[str]`
  - `@dataclass AgentResult: answer: str; steps: list[StepUsage]; tool_names: list[str]; failed: bool; error: str | None`
  - `build_agent(settings: Settings, llm=None)` returns a compiled `create_react_agent` graph (`llm` injectable for tests, default `ChatOllama(model, base_url=ollama_host, temperature=0, num_ctx=...)`).
  - `run_agent(settings, question: str = QUESTION, callbacks: list | None = None, llm=None) -> AgentResult`. Invokes with `config={"callbacks": callbacks or [], "recursion_limit": 12}`.

- [ ] **Step 1: Write failing tests** using a fake chat model (`langchain_core.language_models.fake_chat_models` / `GenericFakeChatModel` extended with `bind_tools` returning self) that scripts three tool calls then a final answer, with `usage_metadata` on each `AIMessage`. Tests: `test_run_collects_steps_and_tool_names` (4 steps, tool names in order `lookup_account`, `get_invoices`, `search_help_center`); `test_recursion_limit_marks_failed` (fake model that always calls a tool: `failed is True`, `error` non-empty, no exception escapes).
- [ ] **Step 2: Run** `uv run pytest tests/test_agent.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Build `StepUsage` from each `AIMessage.usage_metadata` (`input_tokens`, `output_tokens`) in the returned messages. Catch `GraphRecursionError` and any tool/model exception into `failed`/`error`. Missing usage metadata handling lives in Task 4, but `StepUsage` must carry `None`-safe zeros via a helper `usage_of(msg) -> tuple[int, int]` that returns `(0, 0)` and logs a warning when metadata is absent.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: shared langgraph support agent`.

### Task 4: Coarse tracing (Agent A)

**Files:**
- Create: `src/agent_cost_eval/coarse.py`
- Test: `tests/test_coarse.py`

**Interfaces:**
- Consumes: `run_agent`, `AgentResult`, `Settings`.
- Produces: `sum_usage(steps: list[StepUsage]) -> tuple[int, int]` and `run_coarse(settings, run_id: str, llm=None) -> tuple[AgentResult, str]` returning the result and the Langfuse trace id.

- [ ] **Step 1: Write failing tests:** `test_sum_usage` (steps (2100,60), (2600,70), (12500,80), (15800,200) sum to (33000, 410), the slide's numbers); `test_sum_usage_missing_metadata_warns` (steps with zeros produce `(0, 0)` and a caplog warning from `usage_of`).
- [ ] **Step 2: Run** `uv run pytest tests/test_coarse.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `run_coarse`.** Use `langfuse.get_client()`. Open `start_as_current_span(name="support-agent-run")` setting trace tags/metadata `run_id`, `variant=coarse`, then a child `start_as_current_generation(name="support-agent", model=settings.ollama_model)`. Call `run_agent` with **no callbacks**. After it returns, `update(output=answer, usage_details={"input": i, "output": o})` on the generation. Inside the same span, create one empty child span per `result.tool_names` entry named `tool:<name>` with no input/output. Return the trace id from `client.get_current_trace_id()`. On `failed`, set `level="ERROR"` and `status_message=error`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: coarse (summed) tracing agent`.

### Task 5: Granular tracing (Agent B)

**Files:**
- Create: `src/agent_cost_eval/granular.py`
- Test: `tests/test_granular.py`

**Interfaces:**
- Consumes: `run_agent`, `Settings`.
- Produces: `check_payload_visible(steps: list[StepUsage]) -> bool` (True when the step after `get_invoices` has `input_tokens > 8000`, logs a warning with the `num_ctx` hint otherwise) and `run_granular(settings, run_id: str, llm=None) -> tuple[AgentResult, str]`.

- [ ] **Step 1: Write failing tests:** `test_payload_visible_true` (steps where the step following a `get_invoices` call has 12,500 input tokens returns True); `test_payload_visible_false_warns` (3,000 input tokens returns False and caplog contains "num_ctx").
- [ ] **Step 2: Run** `uv run pytest tests/test_granular.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Create `langfuse.langchain.CallbackHandler()`, open `start_as_current_span(name="support-agent")` with tags/metadata `run_id`, `variant=granular`, call `run_agent(..., callbacks=[handler])`, call `check_payload_visible`, return result and trace id. Mark the span `ERROR` on failure.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: granular (per-step) tracing agent`.

### Task 6: Pricing, preflight, Docker and compare CLI

**Files:**
- Create: `docker-compose.yml`, `src/agent_cost_eval/pricing.py`, `src/agent_cost_eval/preflight.py`, `src/agent_cost_eval/compare.py`
- Test: `tests/test_preflight.py`, `tests/test_compare.py`

**Interfaces:**
- Consumes: `Settings`, `run_coarse`, `run_granular`.
- Produces:
  - `register_model_price(settings) -> None` in `pricing.py` (`POST {langfuse_host}/api/public/models` with basic auth, body `modelName`, `matchPattern` (`(?i)^llama3\.1:8b$`, escaped from the setting), `unit="TOKENS"`, `inputPrice`, `outputPrice`); `main()` entry.
  - `preflight(settings) -> None` raising `PreflightError(message)` for: empty Langfuse keys, Langfuse `/api/public/health` unreachable, Ollama `/api/tags` unreachable, model missing from tags (message includes `ollama pull <model>`).
  - `compare.main()`: preflight, `run_id = uuid4().hex[:8]`, run coarse then granular each in try/except so one failure doesn't stop the other, flush client, print both trace URLs (`{host}/trace/{id}`) and a summary (A total tokens; B per-step table).

- [ ] **Step 1: Write failing tests:** in `test_preflight.py` with `requests` mocked (`responses` lib): each of the four failure cases raises `PreflightError` with the expected text; healthy case returns None. In `test_compare.py` with `run_coarse` monkeypatched to raise, assert `main()` still calls `run_granular` and exits non-zero but prints the granular URL.
- [ ] **Step 2: Run** `uv run pytest tests/test_preflight.py tests/test_compare.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules. `docker-compose.yml` is the official Langfuse v3 self-host compose file from the `langfuse/langfuse` repository, pinned to a released tag (record the tag in a comment), with `LANGFUSE_INIT_*` left unset so keys are created in the UI.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: pricing, preflight, docker compose, compare CLI`.

### Task 7: Integration tests and README

**Files:**
- Create: `tests/test_integration.py`, `README.md`

**Interfaces:**
- Consumes: `run_coarse`, `run_granular`, Langfuse public API (`GET /api/public/traces/{id}`, `GET /api/public/observations?traceId=`).

- [ ] **Step 1: Write integration tests** marked `@pytest.mark.integration`, skipped via fixture when `preflight` raises. `test_granular_trace_shape`: after `run_granular`, flush, poll the API (up to 30 s) and assert at least 3 `GENERATION` observations, 3 observations named like `lookup_account`/`get_invoices`/`search_help_center`, and the `get_invoices` output contains `inv_` ids for all 24 invoices. `test_coarse_trace_shape`: exactly 1 `GENERATION`, tool spans have no output. `test_totals_match`: coarse input+output total within 5% of the granular step sum.
- [ ] **Step 2: Run** `uv run pytest -m integration -v` with Ollama and Langfuse up. Expected: PASS. Without them: tests skip.
- [ ] **Step 3: Write `README.md`** with the five-step run guide from the spec, a note that prices are placeholders, and a "what to look for" section (B: biggest input at the step after `get_invoices`; A: only the lump total).
- [ ] **Step 4: Manual verification:** run `uv run python -m agent_cost_eval.compare`, open both URLs, confirm the success criteria in the spec.
- [ ] **Step 5: Commit** `docs: integration tests and README`.
