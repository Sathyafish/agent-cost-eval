"""Agent A: same agent, no per-step tracing. One generation with summed tokens."""
from __future__ import annotations

from langfuse import get_client

from agent_cost_eval.agent import AgentResult, StepUsage, run_agent
from agent_cost_eval.config import Settings


def sum_usage(steps: list[StepUsage]) -> tuple[int, int]:
    return sum(s.input_tokens for s in steps), sum(s.output_tokens for s in steps)


def run_coarse(settings: Settings, run_id: str, llm=None) -> tuple[AgentResult, str | None]:
    client = get_client()
    with client.start_as_current_span(name="support-agent-run") as run_span:
        run_span.update_trace(
            tags=["variant:coarse", f"run:{run_id}"],
            metadata={"run_id": run_id, "variant": "coarse"},
        )
        with client.start_as_current_generation(
            name="support-agent", model=settings.ollama_model
        ) as gen:
            result = run_agent(settings, callbacks=None, llm=llm)  # no handler: no per-step spans
            i, o = sum_usage(result.steps)
            gen.update(output=result.answer, usage_details={"input": i, "output": o})
            if result.failed:
                gen.update(level="ERROR", status_message=result.error)
            # Tool names only: no input, no output, no tokens.
            for name in result.tool_names:
                client.start_span(name=f"tool:{name}").end()
        trace_id = client.get_current_trace_id()
    return result, trace_id
