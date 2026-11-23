"""The one shared support agent. Both tracing modes call run_agent()."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from agent_cost_eval.config import Settings
from agent_cost_eval.tools import TOOLS

log = logging.getLogger(__name__)

QUESTION = "Why was I charged twice this month?"
RECURSION_LIMIT = 12
SYSTEM_PROMPT = (
    "You are a billing support agent. The customer's email is jo@example.com. "
    "Look up their account, review their invoices, check the help center, "
    "then explain clearly what happened and what happens next."
)


@dataclass
class StepUsage:
    step: int
    input_tokens: int
    output_tokens: int
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    steps: list[StepUsage]
    tool_names: list[str]
    failed: bool = False
    error: str | None = None


def usage_of(msg: AIMessage) -> tuple[int, int]:
    """(input, output) tokens for one model response; (0, 0) plus a warning if absent."""
    usage = getattr(msg, "usage_metadata", None)
    if not usage:
        log.warning("No usage metadata on model response; logging 0 tokens for this step")
        return 0, 0
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def build_agent(settings: Settings, llm=None):
    if llm is None:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_host,
            temperature=0,
            num_ctx=settings.num_ctx,
        )
    return create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)


def _summarise(messages: list, failed: bool, error: str | None) -> AgentResult:
    steps: list[StepUsage] = []
    tool_names: list[str] = []
    answer = ""
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        i, o = usage_of(msg)
        names = [c["name"] for c in msg.tool_calls]
        tool_names.extend(names)
        steps.append(StepUsage(len(steps) + 1, i, o, names))
        if not names:
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
    return AgentResult(answer, steps, tool_names, failed, error)


def run_agent(
    settings: Settings,
    question: str = QUESTION,
    callbacks: list | None = None,
    llm=None,
) -> AgentResult:
    graph = build_agent(settings, llm)
    config = {"callbacks": callbacks or [], "recursion_limit": RECURSION_LIMIT}
    messages: list = []
    try:
        for state in graph.stream(
            {"messages": [("user", question)]}, config=config, stream_mode="values"
        ):
            messages = state["messages"]
    except Exception as exc:  # recursion limit, tool or model failure
        return _summarise(messages, True, f"{type(exc).__name__}: {exc}")
    return _summarise(messages, False, None)
