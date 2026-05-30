# HAR Conversation Log Extractor

A small agent that reads a HAR file from a chat session, figures out where the
conversation lives inside all that HTTP traffic, and writes a standalone Python
script that does the same extraction on future HARs from the same provider —
no LLM at extraction time.

The agent itself can run on Claude, OpenAI, or Kimi. It doesn't know which chat
app it's looking at and never sees the prompts you typed.

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Open `.env` and fill in whichever key matches the LLM you want to drive the
agent with:

- `ANTHROPIC_API_KEY` for `--llm-provider claude` (default)
- `OPENAI_API_KEY` for `--llm-provider openai`
- `MOONSHOT_API_KEY` for `--llm-provider kimi`

## 2. Capture the HARs

Any modern browser works — Chrome, Edge, Brave, or Firefox. The key thing is
**Preserve log** (Chrome) / **Persist Logs** (Firefox) so the network panel
keeps entries across the streaming responses.

```
1. Open DevTools → Network tab → enable "Preserve log" (Chrome) or
   "Persist Logs" (Firefox).
2. Go to chatgpt.com (or claude.ai), sign in, send 3 messages, wait for
   each reply to finish before sending the next.
3. Right-click the network panel → "Save all as HAR with content" (Chrome)
   or "Save All As HAR" (Firefox).
4. Save to seed/chatgpt.har or seed/claude.har.
5. Write down the exact text of each message you typed in seed/ground_truth.json:

   {
     "chatgpt": ["first message", "second message", "third message"],
     "claude":  ["first message", "second message", "third message"]
   }
```

The ground truth file is only used after the agent finishes — to check whether
it picked the right endpoint. The agent never sees it.

## 3. Run the agent

```bash
python -m agent seed/chatgpt.har \
    --ground-truth seed/ground_truth.json \
    --provider-name chatgpt \
    --out generated/chatgpt_extractor.py \
    --trace traces/run_01.json
```

That's it. The agent will explore the HAR, write an extractor, run it, check
the structure, fix any issues, and stop when the structural check passes. The
CLI then runs the emitted extractor one more time and diffs the output against
your ground truth. Both results appear in the trace and in the final JSON the
CLI prints.

If you want to run with a different LLM:

```bash
# OpenAI
python -m agent seed/chatgpt.har ... --llm-provider openai --model gpt-4o

# Kimi (Moonshot)
python -m agent seed/chatgpt.har ... --llm-provider kimi --model kimi-k2-0905-preview
```

## 4. Use the emitted extractor

The script the agent writes is plain Python — no `anthropic`, no `openai`, no
`requests`. You can run it on any HAR from the same provider:

```bash
python generated/chatgpt_extractor.py --har seed/chatgpt.har --out logs/chatgpt.json
```

The output is a JSON array, one entry per turn:

```json
[
  {"turn": 0, "role": "user", "content": "hi", "endpoint": "https://...", "timestamp": 1710000000},
  {"turn": 1, "role": "assistant", "content": "hello!", "endpoint": "https://...", "timestamp": 1710000001}
]
```

## 5. Verify by hand

```bash
python -m agent.verify logs/chatgpt.json --ground-truth seed/ground_truth.json --provider chatgpt
pytest tests/
```

The pytest harness picks up every `generated/*_extractor.py` next to its
matching `seed/*.har`, runs each, and verifies against ground truth.

---

## What's in the repo

```
agent/                  the agent code — no provider knowledge anywhere here
  prompts.py            system prompt (grep this for "chatgpt"/"claude" — nothing)
  tools.py              the 9 tools the agent uses to explore the HAR
  har.py                HAR 1.2 helpers (base64 / gzip / brotli decode)
  llm.py                provider abstraction: Claude, OpenAI, Kimi
  loop.py               tool-use loop, LLM-agnostic
  sandbox.py            subprocess runner + AST import lint for emitted code
  verify.py             ground-truth diff + structural checks
  tracer.py             writes the JSON trace files
  cli.py / __main__.py  python -m agent
generated/              standalone extractors the agent wrote
logs/                   extractor outputs
traces/                 full agent traces — one per run
seed/                   HARs and ground_truth.json
tests/                  pytest for the generated extractors
```

`APPROACH.md` is a short narrative of how I thought about this.
`QUALITY.md` is the proof-it-actually-works document.

## Reading the traces

Each run lands a single JSON file in `traces/`. Open it and you can replay the
agent step by step — every tool call, every result preview, every model
thought. The failed runs are honestly the most interesting ones; that's where
you see the agent picking a wrong endpoint, looking at the body, realizing it's
analytics, and going back to `list_entries`.

| event kind | what it is |
|---|---|
| `run_start` | initial inputs (HAR path, model). No ground truth — the agent doesn't get it. |
| `system_prompt` | the provider-agnostic system prompt |
| `tool_call` / `tool_result` | every tool the agent used and what came back |
| `model_thinking` | summarized adaptive thinking (Claude only) |
| `model_text` | what the agent said out loud between tool calls |
| `finalize_observed` | the agent declared done with a note |
| `run_end` | summary of the run |
| `post_hoc_ground_truth_verify` | the CLI's check against `ground_truth.json`, run *after* the agent finishes |
