"""CLI entry point for the HAR-to-extractor agent."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

from agent.llm import make_provider
from agent.loop import DEFAULT_MAX_TURNS, DEFAULT_MAX_TOKENS, run_agent
from agent.sandbox import run_extractor as run_extractor_sandbox
from agent.tracer import Tracer
from agent.verify import verify as verify_against_ground_truth


_PROVIDER_ENV_VAR = {
    "claude": "ANTHROPIC_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
}


def main(argv: list[str] | None = None) -> int:
    if load_dotenv is not None:
        load_dotenv()

    ap = argparse.ArgumentParser(
        description="Agent: HAR -> standalone Python extractor.",
    )
    ap.add_argument("har", help="path to seed/<provider>.har")
    ap.add_argument("--ground-truth", required=True, help="path to seed/ground_truth.json")
    ap.add_argument(
        "--provider-name",
        required=True,
        help="key into ground_truth.json (your label for this provider) — used for selecting prompts and naming outputs",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="path where the generated extractor will be written (e.g. generated/<provider>_extractor.py)",
    )
    ap.add_argument("--trace", required=True, help="path for the JSON trace file (e.g. traces/run_01.json)")
    ap.add_argument(
        "--llm-provider",
        default="claude",
        choices=["claude", "anthropic", "openai", "gpt", "kimi", "moonshot"],
        help="which LLM drives the agent loop (default: claude)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="override the LLM model id (default: provider-specific flagship)",
    )
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = ap.parse_args(argv)

    required_env = _PROVIDER_ENV_VAR[args.llm_provider]
    if not os.environ.get(required_env):
        print(
            f"ERROR: {required_env} is not set (put it in .env or export it). "
            f"Required for --llm-provider {args.llm_provider}.",
            file=sys.stderr,
        )
        return 2

    gt_raw = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    if isinstance(gt_raw, dict):
        prompts = gt_raw.get(args.provider_name)
        if prompts is None:
            print(
                f"ERROR: provider key {args.provider_name!r} not found in {args.ground_truth}. "
                f"Available keys: {list(gt_raw.keys())}",
                file=sys.stderr,
            )
            return 2
    elif isinstance(gt_raw, list):
        prompts = gt_raw
    else:
        print(f"ERROR: {args.ground_truth} must be a list of prompts or a dict keyed by provider name.", file=sys.stderr)
        return 2

    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        print(f"ERROR: ground truth prompts for {args.provider_name!r} must be a list of strings.", file=sys.stderr)
        return 2

    try:
        llm = make_provider(args.llm_provider)
    except ImportError as e:
        print(f"ERROR: missing SDK for --llm-provider {args.llm_provider}: {e}", file=sys.stderr)
        print("Install with: pip install -e .", file=sys.stderr)
        return 2

    tracer = Tracer(args.trace)
    outcome = run_agent(
        har_path=args.har,
        extractor_out_path=args.out,
        tracer=tracer,
        provider=llm,
        model=args.model,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
    )

    # Post-hoc ground-truth check: the agent finished without ever seeing
    # the typed prompts. Run the emitted extractor and compare its output
    # against ground_truth.json. This result is for the human reviewer; the
    # agent already declared done (or didn't) based on structural checks
    # alone. Trace the result too so it shows up in traces/.
    gt_result = None
    if outcome.get("finalized") and outcome.get("extractor_path"):
        run = run_extractor_sandbox(args.out, args.har)
        if run.ok and isinstance(run.output_json, list):
            v = verify_against_ground_truth(run.output_json, prompts)
            gt_result = v.to_dict()
        else:
            gt_result = {"ok": False, "error": run.error or "extractor failed in post-hoc run"}
        tracer.event("post_hoc_ground_truth_verify", **(gt_result or {}))

    final = {**outcome, "ground_truth_verify": gt_result}
    print(json.dumps(final, indent=2))
    overall_ok = bool(outcome.get("finalized")) and bool((gt_result or {}).get("ok"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
