#!/usr/bin/env python3
"""Extract a chat log from a claude.ai HAR capture."""
import argparse, json, re
from pathlib import Path
from datetime import datetime


def parse_ts(s: str) -> int:
    # ISO 8601 -> epoch seconds
    s = s.replace("Z", "+00:00")
    return int(datetime.fromisoformat(s).timestamp())


def parse_sse(body: str):
    """Yield (event, data_obj) tuples from an SSE body."""
    for block in re.split(r"\r?\n\r?\n", body):
        event = None
        data_parts = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_parts.append(line[5:].lstrip())
        if not data_parts:
            continue
        raw = "\n".join(data_parts)
        try:
            yield event, json.loads(raw)
        except json.JSONDecodeError:
            continue


def reconstruct_text(body: str) -> str:
    out = []
    for _evt, data in parse_sse(body):
        if not isinstance(data, dict):
            continue
        if data.get("type") == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                t = delta.get("text")
                if isinstance(t, str):
                    out.append(t)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    har = json.loads(Path(args.har).read_text(encoding="utf-8"))
    entries = har.get("log", {}).get("entries", [])

    turns = []
    for ent in entries:
        req = ent.get("request", {})
        resp = ent.get("response", {})
        url = req.get("url", "")
        method = req.get("method", "")
        if method != "POST" or not url.endswith("/completion"):
            continue

        # request body has the user prompt
        post = req.get("postData") or {}
        req_text = post.get("text") or ""
        try:
            req_json = json.loads(req_text)
        except json.JSONDecodeError:
            continue
        prompt = req_json.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            continue

        # response body has the streamed assistant text
        resp_text = (resp.get("content") or {}).get("text") or ""
        assistant_text = reconstruct_text(resp_text)

        started = ent.get("startedDateTime") or ""
        try:
            ts = parse_ts(started)
        except Exception:
            ts = 0

        turns.append({
            "role": "user",
            "content": prompt,
            "endpoint": url,
            "timestamp": ts,
        })
        if assistant_text:
            turns.append({
                "role": "assistant",
                "content": assistant_text,
                "endpoint": url,
                "timestamp": ts,
            })

    # ensure ordering by timestamp (entries already in order)
    out = []
    for i, t in enumerate(turns):
        out.append({"turn": i, **t})

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
