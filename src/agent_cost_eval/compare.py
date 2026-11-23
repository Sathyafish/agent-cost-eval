"""Run the same question through Agent A (coarse) and Agent B (granular)."""
from __future__ import annotations

import sys
import uuid

from langfuse import get_client

from agent_cost_eval.coarse import run_coarse, sum_usage
from agent_cost_eval.config import load_settings
from agent_cost_eval.granular import run_granular
from agent_cost_eval.preflight import PreflightError, preflight


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
    results = {}
    for label, fn in (("A (coarse)", run_coarse), ("B (granular)", run_granular)):
        try:
            result, trace_id = fn(settings, run_id)
        except Exception as exc:  # one agent failing must not stop the other
            failed = True
            print(f"Agent {label} FAILED: {type(exc).__name__}: {exc}")
            continue
        results[label] = result
        failed = failed or result.failed
        status = f"FAILED ({result.error})" if result.failed else "ok"
        print(f"Agent {label}: {status}\n  trace: {settings.langfuse_host}/trace/{trace_id}")

    get_client().flush()

    if "A (coarse)" in results:
        i, o = sum_usage(results["A (coarse)"].steps)
        print(f"\nA total: in {i:,} · out {o:,}  (no breakdown)")
    if "B (granular)" in results:
        print("\nB per step:")
        for s in results["B (granular)"].steps:
            tools = ", ".join(s.tool_calls) or "respond"
            print(f"  step {s.step}: in {s.input_tokens:>6,} · out {s.output_tokens:>4,}  [{tools}]")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
