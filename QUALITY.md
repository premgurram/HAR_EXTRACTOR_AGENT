# Quality and Efficiency

How I know this works, and how you can check.

## 1. The agent has no provider knowledge

To verify:

```bash
grep -nirE 'chatgpt|claude\.ai|openai|backend-api|messages\.parts|conversation_id' agent/
```

What you'll see:

- `agent/sandbox.py` mentions `openai` in `FORBIDDEN_NAMES` — that's the
  allowlist explicitly blocking that import in emitted extractors. Same
  for `anthropic`, `httpx`, etc.
- `agent/cli.py` has the word `chatgpt`/`claude` only as example labels in
  argparse help text. The agent itself never sees that string — it's the
  user's choice of provider name for the ground-truth file key.

That's it. No URLs, no field names, no provider names anywhere in the prompt
or the tools.

The system prompt (`agent/prompts.py`) describes HAR file structure (HAR 1.2,
SSE, NDJSON, JSON, base64-encoded bodies) and the output schema. Nothing
provider-specific.

## 2. The agent never sees the ground truth

The ground-truth file is the most tempting place to leak hints. I went through
three iterations to land on the current boundary.

- **Iteration 1.** `read_ground_truth` was a tool, and the agent used the
  typed prompts as search seeds in `search_bodies`. Fast — converged in 3
  or 4 turns — but the agent wasn't really discovering; it was being told
  what to find.
- **Iteration 2.** I removed the tool but kept ground-truth comparison
  inside the verify call, which still returned a `matched X of Y` signal.
  Better, but the agent could still iterate against that signal.
- **Iteration 3 (current).** The agent's tool is `verify_structure`, which
  does only structural checks. Ground-truth comparison happens in the CLI
  *after* the agent has called `finalize`. The agent's `ToolContext`
  doesn't even carry the ground-truth field.

Defense in depth:

- `tools.py` — `ToolContext` has no `ground_truth` field.
- `loop.py` — `run_agent` takes no `ground_truth` parameter.
- `cli.py` — loads `ground_truth.json` but passes nothing about it into
  `run_agent`. The check happens after the agent returns.

Smoke test:

```python
out = tools._tool_verify_structure(ctx, {})
# {'ok': True, 'structural_issues': [], 'summary': {'turns': 2, 'user_turns': 1, 'assistant_turns': 1}}
assert 'ground_truth_diff' not in out
assert not any(k.startswith('ground_truth') for k in out.get('summary', {}))
```

## 3. The emitted extractor cannot call an LLM or the network

Two layers:

- **AST lint** before each run. `agent/sandbox.py` parses the extractor
  source and rejects any `import` not in the standard-library allowlist.
  Explicitly blocked: `anthropic`, `openai`, `requests`, `httpx`,
  `aiohttp`, `urllib3`, `urllib.request`, `socket`, `http.client`,
  `subprocess`, `os.system`.
- **Subprocess execution** with a minimal `env`. The extractor process
  inherits no `ANTHROPIC_API_KEY`, no `OPENAI_API_KEY`, no
  `MOONSHOT_API_KEY`. Even if it somehow loaded an SDK, it has no credentials.

The lint runs at `write_extractor` time *and* at `run_extractor` time, so an
extractor that fails the lint never even executes.

## 4. Verification is multi-layered

| Where | What it checks | Who sees it |
|---|---|---|
| `verify_structure` tool | turn indexing, role alternation, non-empty content, monotonic epoch-shaped timestamps, valid http(s) endpoints, ≥ 2 turns | the agent (during the loop) |
| `cli.py` post-hoc | runs the emitted extractor, diffs the output against `seed/ground_truth.json` | the human (after the run) |
| `python -m agent.verify` | standalone version of the same diff | the human (any time) |
| `pytest tests/` | runs every `generated/*_extractor.py` against its matching `seed/*.har` | CI / a reviewer |

A structurally valid extractor that picked the wrong endpoint will pass the
in-loop check and fail the post-hoc check. That failure is logged into the
trace as `post_hoc_ground_truth_verify`, surfaced in the CLI's final JSON
under `ground_truth_verify`, and reflected in the CLI's exit code (`0` only
if both `finalized` and `ground_truth_verify.ok` are true).

## 5. Efficiency

The agent's wall-clock cost on a single HAR is bounded by the turn cap
(default 25) and the max-tokens-per-turn cap (default 8192). In practice it
finishes in 5–10 turns on a clean HAR.

Token usage is kept down by a few specific design choices:

- **Score-based ranking** in `list_entries`. The agent doesn't have to walk
  100 entries; the top 3–5 by score are almost always the right candidates.
  The score adds points for POST + JSON/SSE/NDJSON + reasonable size +
  non-trivial request body, and subtracts for telemetry-looking URLs.
- **Slice sampling** for large bodies. `sample_body_slices` returns head /
  middle / tail of ~1500 bytes each rather than dumping a 50KB SSE stream
  into context. The agent sees real content chunks, not just headers.
- **Prompt caching** (Claude only). System prompt and the (stable) last
  tool definition get `cache_control: ephemeral`. A growing conversation
  gets a moving cache breakpoint on the last user message so the prefix
  caches incrementally turn-over-turn. The trace's `model_response` events
  include `cache_read_input_tokens` so you can see the cache working.
- **Adaptive thinking with summarized display** (Claude only). The model
  decides when to think; the trace shows the summary so reviewers can see
  reasoning without paying for full omitted blocks.

## 6. Honest limitations

- **Single-HAR scope.** Each run handles one HAR. Multi-conversation HARs
  with mixed providers aren't tested.
- **The agent can pick a structurally valid wrong endpoint.** This is the
  whole point of splitting verify — the post-hoc check is what catches
  this case. It does catch it (exit code goes non-zero), but only after
  the agent finished.
- **No tests for the agent loop itself.** Per the requirement. The traces are the
  primary record.

## 7. Traces

Reading the traces is the most useful thing a reviewer can do. They're
checked-in JSON, ordered events, one file per run. The events you'll find
useful:

- `tool_call` / `tool_result` — every tool the agent used, with truncated
  result previews so you can see what the model saw.
- `model_thinking` — summarized reasoning between tool calls.
- `model_text` — the agent's narration to itself.
- `post_hoc_ground_truth_verify` — the final check, always present at
  the end of a finalized run.

Look especially for traces where `verify_structure` returned issues that
caused the agent to rewrite the extractor — that's where the loop earns
its keep.
