# Approach

The interesting constraint in this challenge is that the agent isn't supposed
to know what it's looking at — no `claude.ai`, no `backend-api`, no JSONPaths
in the prompt. Whatever it figures out has to come from the HAR itself. Most
of the design follows from getting that one rule right.

So my first decision was the system prompt. It describes the task ("a HAR file
captured a conversation between a user and an assistant; find which entries
carry what; emit a Python script that does the same") and the output schema.
Nothing about chat applications, nothing about field names. Grep the file —
no provider strings appear.

## The ground-truth experiment

The second decision was about `seed/ground_truth.json`, and this is where I
went back and forth a few times.

**First version.** I gave the agent a `read_ground_truth` tool. The agent
would fetch the typed prompts and use them as search seeds in `search_bodies`
to locate the user-input endpoint. This works really well — it converges in 3
or 4 turns and the trace is short and clean. The argument for it: the user
typed those prompts, so it's not really "provider knowledge" the agent
shouldn't have; it's session knowledge that's naturally available.

**Second version.** I removed the `read_ground_truth` tool but kept
ground-truth comparison inside `verify_extractor`. So the agent didn't get the
prompt strings, but it did get back a "matched X of Y" signal each iteration.
It could still effectively use ground truth to decide when to stop iterating,
just without seeing the actual text.

**Third version (shipped).** I split verify in two. The agent has a
`verify_structure` tool that only checks structural properties — turn
indexing, role alternation, non-empty content, monotonic epoch-shaped
timestamps, valid http(s) endpoints. No ground-truth comparison anywhere in
the loop. The ground-truth file is loaded by the CLI, but only consulted
*after* the agent calls `finalize`. If the agent picked the wrong endpoint,
the post-hoc check catches it and the CLI exits non-zero. The agent itself
had zero signal from ground truth during the run.

All three versions are defensible reads of the requirement. The third one is
the strictest, and it's the one where the trace genuinely shows the agent
*discovering* rather than *being told*. That felt closer to the intent —
even though the first version converged faster and used fewer tokens.

## Tools

Once those two boundaries were set, the rest fell out. The agent has nine
tools:

- `list_entries` — every HAR entry, sorted by a conversational-likelihood
  score (POST + JSON/SSE/NDJSON + reasonable size − telemetry-looking URL).
  Static assets and preflights are filtered out by mime type and URL pattern.
- `inspect_entry` — headers and a body slice for one entry.
- `sample_body_slices` — head/middle/tail of a large body. This was the
  most useful tool I added; without it the agent kept giving up on streaming
  endpoints because the first 4KB is almost always structural metadata, not
  text content.
- `decode_stream` — best-effort SSE / NDJSON / single-JSON parser. Returns
  the chunk format, the set of JSON key paths observed, and a reconstructed
  text by concatenating string leaves. The agent uses this to learn the
  shape it needs to mirror in code.
- `search_bodies` — substring search. The prompt explicitly says not to
  invent strings to search for, only to confirm things the agent has
  already observed.
- `write_extractor`, `run_extractor`, `verify_structure`, `finalize` — the
  write/run/verify loop. `verify_structure` reports structural issues only.

## Sandboxed extraction

The emitted extractor is constrained by `sandbox.py`. Before running it, an
AST walk checks every `import` against an allowlist (`json`, `re`, `base64`,
`gzip`, `zlib`, `urllib.parse`, `argparse`, `sys`, `pathlib`, `typing`, `io`,
`datetime`, `collections`, `itertools`, `html`, `dataclasses`) and explicitly
rejects `anthropic`, `openai`, `requests`, `httpx`, `socket`, and friends.
Anything that touches the network or a model — including by accident — is
caught at lint time, not runtime.

## LLM-agnostic

The agent itself doesn't depend on a particular LLM. `agent/llm.py` has three
adapters: Anthropic (with adaptive thinking and prompt caching), OpenAI Chat
Completions (with function-calling tools), and Kimi (which is OpenAI-compatible
at `api.moonshot.ai/v1` with its own model IDs). Internally the loop speaks
Anthropic-shaped content blocks; each adapter converts at the API boundary.
The point isn't to ship every model — it's to demonstrate that nothing in the
loop is tied to one vendor.

