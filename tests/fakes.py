"""Test-only scripted chat model with usage metadata and bind_tools support."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def ai(tool: str | None, inp: int, out: int, content: str = "") -> AIMessage:
    calls = (
        [{"name": tool, "args": _ARGS[tool], "id": f"call_{tool}", "type": "tool_call"}]
        if tool
        else []
    )
    return AIMessage(
        content=content,
        tool_calls=calls,
        usage_metadata={"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out},
    )


_ARGS = {
    "lookup_account": {"email": "jo@example.com"},
    "get_invoices": {"account_id": "acct_1"},
    "search_help_center": {"query": "charged twice"},
}


class ScriptedModel(BaseChatModel):
    """Replays `script`; if `loop` is true, repeats the last message forever."""

    script: list[Any]
    loop: bool = False
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        idx = min(self.i, len(self.script) - 1) if self.loop else self.i
        msg = self.script[idx]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def slide_script() -> list[AIMessage]:
    """The reference slide's numbers."""
    return [
        ai("lookup_account", 2100, 60),
        ai("get_invoices", 2600, 70),
        ai("search_help_center", 12500, 80),
        ai(None, 15800, 200, content="You were charged twice because of a retried payment."),
    ]
