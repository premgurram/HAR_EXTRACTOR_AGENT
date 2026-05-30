"""Verify an extracted conversation log against ground truth + structural rules.

Provider-agnostic. Knows only the output schema:
    [{"turn": int, "role": "user"|"assistant", "content": str,
      "endpoint": str, "timestamp": int}]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


URL_RE = re.compile(r"^https?://", re.IGNORECASE)


@dataclass
class VerifyResult:
    ok: bool
    ground_truth_diff: list[dict[str, Any]] = field(default_factory=list)
    structural_issues: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify(log: list[dict[str, Any]], ground_truth_prompts: list[str] | None) -> VerifyResult:
    issues: list[str] = []
    diff: list[dict[str, Any]] = []

    if not isinstance(log, list):
        return VerifyResult(False, [], [f"log is not a list (got {type(log).__name__})"])

    if not log:
        return VerifyResult(False, [], ["log is empty"])

    expected_role = "user"
    last_ts: int | None = None
    user_turns: list[dict[str, Any]] = []
    for i, t in enumerate(log):
        if not isinstance(t, dict):
            issues.append(f"turn {i}: not an object")
            continue
        for k in ("turn", "role", "content", "endpoint", "timestamp"):
            if k not in t:
                issues.append(f"turn {i}: missing key '{k}'")

        if t.get("turn") != i:
            issues.append(f"turn {i}: 'turn' field is {t.get('turn')!r}, expected {i}")

        role = t.get("role")
        if role not in ("user", "assistant"):
            issues.append(f"turn {i}: role {role!r} not in ('user','assistant')")
        elif role != expected_role:
            issues.append(f"turn {i}: role {role!r} breaks alternation (expected {expected_role!r})")
        expected_role = "assistant" if expected_role == "user" else "user"

        content = t.get("content")
        if not isinstance(content, str) or not content.strip():
            issues.append(f"turn {i}: content is empty or not a string")

        endpoint = t.get("endpoint", "")
        if not isinstance(endpoint, str) or not URL_RE.match(endpoint):
            issues.append(f"turn {i}: endpoint {endpoint!r} is not a valid http(s) URL")

        ts = t.get("timestamp")
        if not isinstance(ts, int):
            issues.append(f"turn {i}: timestamp {ts!r} is not an int")
        else:
            # Accept 10-digit (seconds) or 13-digit (ms) epoch.
            digits = len(str(ts))
            if digits not in (10, 13):
                issues.append(f"turn {i}: timestamp {ts} doesn't look like epoch s/ms ({digits} digits)")
            if last_ts is not None and ts < last_ts:
                issues.append(f"turn {i}: timestamp {ts} earlier than previous {last_ts}")
            last_ts = ts

        if role == "user":
            user_turns.append(t)

    if ground_truth_prompts:
        # Match in order: each ground-truth prompt must appear as a substring of
        # the next user turn's content. Whitespace is normalized.
        gt = [_norm(p) for p in ground_truth_prompts]
        user_contents = [_norm(str(t.get("content", ""))) for t in user_turns]
        ui = 0
        for pi, prompt in enumerate(gt):
            matched_at = None
            while ui < len(user_contents):
                if prompt in user_contents[ui] or user_contents[ui] in prompt:
                    matched_at = ui
                    ui += 1
                    break
                ui += 1
            if matched_at is None:
                diff.append({"prompt_index": pi, "prompt": ground_truth_prompts[pi], "matched": False})
            else:
                diff.append({"prompt_index": pi, "prompt": ground_truth_prompts[pi], "matched": True, "user_turn": matched_at})

    summary = {
        "turns": len(log),
        "user_turns": sum(1 for t in log if isinstance(t, dict) and t.get("role") == "user"),
        "assistant_turns": sum(1 for t in log if isinstance(t, dict) and t.get("role") == "assistant"),
        "ground_truth_matched": sum(1 for d in diff if d.get("matched")),
        "ground_truth_total": len(diff),
    }

    ok = not issues and all(d.get("matched") for d in diff)
    return VerifyResult(ok=ok, ground_truth_diff=diff, structural_issues=issues, summary=summary)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a conversation log.")
    ap.add_argument("log", help="path to logs/<provider>.json")
    ap.add_argument("--ground-truth", required=False, help="path to seed/ground_truth.json")
    ap.add_argument("--provider", required=False, help="key inside ground_truth.json (your label for this provider)")
    args = ap.parse_args(argv)

    log = json.loads(Path(args.log).read_text(encoding="utf-8"))
    prompts: list[str] | None = None
    if args.ground_truth:
        gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
        if isinstance(gt, dict) and args.provider:
            prompts = gt.get(args.provider)
        elif isinstance(gt, list):
            prompts = gt

    result = verify(log, prompts)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
