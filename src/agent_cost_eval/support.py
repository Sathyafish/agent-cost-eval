"""The traced customer support agent: one Langfuse trace per customer question.

Langfuse's LangChain CallbackHandler records an agent span, one generation per
reasoning step (with its own tokens and cost) and one span per tool call.
"""
from __future__ import annotations

import logging

from langfuse import get_client, propagate_attributes
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
    return False


def run_support_agent(
    settings: Settings,
    question: str,
    user_id: str,
    session_id: str,
    llm=None,
) -> tuple[AgentResult, str | None]:
    """Answer one question; returns the result and the Langfuse trace id."""
    client = get_client()
    handler = CallbackHandler()
    with propagate_attributes(user_id=user_id, session_id=session_id, tags=["support-agent"]):
        with client.start_as_current_span(name="support-agent", input=question) as span:
            result = run_agent(settings, question=question, callbacks=[handler], llm=llm)
            span.update(output=result.answer)
            if result.failed:
                span.update(level="ERROR", status_message=result.error)
            else:
                check_payload_visible(result.steps)
            trace_id = client.get_current_trace_id()
    return result, trace_id
