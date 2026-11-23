"""Send sample customer questions through the support agent so Langfuse has data to show."""
from __future__ import annotations

import sys
import uuid

from langfuse import get_client

from agent_cost_eval.config import load_settings
from agent_cost_eval.preflight import PreflightError, preflight
from agent_cost_eval.support import run_support_agent

# (user_id, question). Several users so cost-per-user is meaningful on the dashboard.
SAMPLE_QUESTIONS = [
    ("user-alice", "Why was I charged twice this month?"),
    ("user-alice", "Can I get a refund for the duplicate charge?"),
    ("user-bob", "What payment method is on my account?"),
    ("user-carol", "Why was I charged twice this month?"),
    ("user-bob", "How do refunds work and how long do they take?"),
]


def main() -> int:
    settings = load_settings()
    try:
        preflight(settings)
    except PreflightError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2

    run_id = uuid.uuid4().hex[:8]
    print(f"run_id={run_id}  model={settings.ollama_model}")
    failed = False
    for n, (user_id, question) in enumerate(SAMPLE_QUESTIONS, start=1):
        session_id = f"{run_id}-{n}"
        try:
            result, trace_id = run_support_agent(settings, question, user_id, session_id)
        except Exception as exc:  # one question failing must not stop the rest
            failed = True
            print(f"[{n}] {user_id}: FAILED {type(exc).__name__}: {exc}")
            continue
        failed = failed or result.failed
        tokens_in = sum(s.input_tokens for s in result.steps)
        tokens_out = sum(s.output_tokens for s in result.steps)
        status = f"FAILED ({result.error})" if result.failed else "ok"
        print(
            f"[{n}] {user_id}: {status}  in {tokens_in:,} · out {tokens_out:,}\n"
            f"    {question}\n    trace: {settings.langfuse_public_url}/trace/{trace_id}"
        )
    get_client().flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
