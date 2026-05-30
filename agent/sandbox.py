"""Run a generated extractor in a subprocess with a timeout, and lint its imports.

The extractor must be pure Python — no LLM, no network. We enforce this two ways:
  1. AST walk over the generated source to reject disallowed imports.
  2. Subprocess execution with no inherited env beyond a minimal safe set.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ALLOWED_IMPORTS = {
    "json",
    "re",
    "base64",
    "gzip",
    "zlib",
    "urllib.parse",
    "urllib",
    "argparse",
    "sys",
    "pathlib",
    "typing",
    "io",
    "datetime",
    "collections",
    "itertools",
    "html",
    "dataclasses",
}

FORBIDDEN_NAMES = {
    "anthropic",
    "openai",
    "requests",
    "httpx",
    "aiohttp",
    "urllib.request",
    "urllib3",
    "socket",
    "http.client",
    "subprocess",
    "os.system",
}


@dataclass
class LintResult:
    ok: bool
    issues: list[str]


def lint_extractor_source(source: str) -> LintResult:
    issues: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return LintResult(False, [f"SyntaxError: {e}"])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                _check_module(n.name, issues)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            _check_module(mod, issues)

    return LintResult(not issues, issues)


def _check_module(name: str, issues: list[str]) -> None:
    if not name:
        return
    if name in FORBIDDEN_NAMES:
        issues.append(f"disallowed import: {name}")
        return
    top = name.split(".")[0]
    if name not in ALLOWED_IMPORTS and top not in ALLOWED_IMPORTS:
        issues.append(f"non-allowlisted import: {name}")


@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    output_json: list | dict | None
    error: str | None


def run_extractor(extractor_path: str | Path, har_path: str | Path, timeout: float = 30.0) -> RunResult:
    extractor_path = Path(extractor_path)
    har_path = Path(har_path)

    if not extractor_path.exists():
        return RunResult(False, "", "", None, f"extractor not found: {extractor_path}")
    if not har_path.exists():
        return RunResult(False, "", "", None, f"HAR not found: {har_path}")

    src = extractor_path.read_text(encoding="utf-8")
    lint = lint_extractor_source(src)
    if not lint.ok:
        return RunResult(False, "", "", None, "lint failed: " + "; ".join(lint.issues))

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "out.json"
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            # Deliberately not passing ANTHROPIC_API_KEY / OPENAI_API_KEY etc.
        }
        try:
            proc = subprocess.run(
                [sys.executable, str(extractor_path), "--har", str(har_path), "--out", str(out_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            return RunResult(False, e.stdout or "", e.stderr or "", None, f"timeout after {timeout}s")

        if proc.returncode != 0:
            return RunResult(False, proc.stdout, proc.stderr, None, f"exit code {proc.returncode}")

        if not out_path.exists():
            return RunResult(False, proc.stdout, proc.stderr, None, "extractor produced no output file")

        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return RunResult(False, proc.stdout, proc.stderr, None, f"invalid JSON output: {e}")

        return RunResult(True, proc.stdout, proc.stderr, data, None)
