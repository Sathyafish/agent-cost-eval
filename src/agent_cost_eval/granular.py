"""Agent B: same agent, traced per step via Langfuse's LangChain CallbackHandler."""
from __future__ import annotations

import logging

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agent_cost_eval.agent import AgentResult, StepUsage, run_agent
from agent_cost_eval.config import Settings

log = logging.getLogger(__name__)
PAYLOAD_MIN_INPUT_TOKENS = 8000


def check_payload_visible(steps: list[StepUsage]) -> bool:
    """True if the step after get_invoices carried the big payload (input > 8,000 tokens)."""
    for prev, nxt in zip(steps, steps[1:]):
        if "get_invoices" in prev.tool_calls:
            if nxt.input_tokens > PAYLOAD_MIN_INPUT_TOKENS:
                return True
            log.warning(
                "Step after get_invoices has only %d input tokens; the invoice payload "
                "was probably truncated. Raise num_ctx (NUM_CTX in .env).",
                nxt.input_tokens,
            )
            return False
    log.warning("get_invoices was never called; cannot check payload visibility")
    return False


def run_granular(settings: Settings, run_id: str, llm=None) -> tuple[AgentResult, str | None]:
    client = get_client()
    handler = CallbackHandler()
    with client.start_as_current_span(name="support-agent") as span:
        span.update_trace(
            tags=["variant:granular", f"run:{run_id}"],
            metadata={"run_id": run_id, "variant": "granular"},
        )
        result = run_agent(settings, callbacks=[handler], llm=llm)
        if result.failed:
            span.update(level="ERROR", status_message=result.error)
        else:
            check_payload_visible(result.steps)
        trace_id = client.get_current_trace_id()
    return result, trace_id
