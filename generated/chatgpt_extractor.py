#!/usr/bin/env python3
"""Extract a chat log from a ChatGPT HAR capture.

The HAR records POST requests to https://chatgpt.com/backend-api/f/conversation
with `Content-Type: application/json` and `accept: text/event-stream`.

Each such entry carries:
  - one USER turn in the JSON request body:
      messages[0].content.parts[0]  (text)
      messages[0].create_time       (float epoch seconds)
  - one ASSISTANT turn streamed back as SSE using the
    delta_encoding "v1" patch format:
      * Each delta event's data is JSON.
      * `{"p": "", "o": "add", "v": {"message": {...}, ...}}` introduces a
        message object; the most recently added message is the active target
        for subsequent path-based ops.
      * `{"p": "/message/content/parts/0", "o": "append", "v": "<txt>"}`
        appends text to that message's part 0.
      * `{"o": "patch", "v": [<op>, ...]}` or shorthand `{"v": [<op>,...]}`
        applies a batch of such ops.
      * Shorthand `{"v": "<txt>"}` continues the previous (path, op).
    The visible answer is the message with author.role == "assistant",
    recipient == "all", and content.content_type == "text".
"""

import argparse
import base64
import json
import re
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers


def parse_iso8601(s: str) -> int:
    """Return epoch seconds for an ISO-8601 timestamp from a HAR entry."""
    if not s:
        return 0
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except Exception:
        return 0


def get_body_text(message: dict) -> str:
    """Return the (possibly base64-decoded) body text for a HAR request/response."""
    content = (message or {}).get("content") if "content" in (message or {}) else None
    if content is None:
        # request shape
        pd = (message or {}).get("postData") or {}
        return pd.get("text") or ""
    text = content.get("text") or ""
    if content.get("encoding") == "base64" and text:
        try:
            text = base64.b64decode(text).decode("utf-8", errors="replace")
        except Exception:
            pass
    return text


# ---------------------------------------------------------------------------
# SSE parsing


def parse_sse(body: str):
    """Yield (event_name_or_None, data_payload_str) for each SSE block."""
    # Blocks are separated by blank lines.
    for block in re.split(r"\n\n+", body):
        ev = None
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[len("event:"):].strip()
            elif line.startswith("data:"):
                # SSE: strip one leading space if present
                data_lines.append(line[len("data:"):].lstrip(" "))
        if data_lines:
            yield ev, "\n".join(data_lines)


def reconstruct_assistant(body: str) -> str:
    """Reconstruct the assistant's visible answer text from an SSE body."""
    messages = {}            # id -> {role, recipient, content_type, parts0}
    current_id = None        # most recently added message id
    last_path_op = None      # (path, op) tuple, for shorthand {"v": ...}

    def register(msg_wrap):
        nonlocal current_id
        msg = (msg_wrap or {}).get("message")
        if not isinstance(msg, dict):
            return
        mid = msg.get("id")
        if not mid:
            return
        author = msg.get("author") or {}
        content = msg.get("content") or {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        first = parts[0] if parts and isinstance(parts[0], str) else ""
        messages[mid] = {
            "role": author.get("role"),
            "recipient": msg.get("recipient"),
            "content_type": content.get("content_type"),
            "parts0": first,
        }
        current_id = mid

    def apply_op(p, o, v):
        nonlocal last_path_op
        if isinstance(p, str) and isinstance(o, str):
            last_path_op = (p, o)
        if (
            p == "/message/content/parts/0"
            and o == "append"
            and isinstance(v, str)
            and current_id in messages
        ):
            messages[current_id]["parts0"] += v

    for _ev, data in parse_sse(body):
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        # Skip framing / metadata events: they have a string "type" key.
        if isinstance(obj.get("type"), str):
            continue

        p = obj.get("p")
        o = obj.get("o")
        v = obj.get("v")

        # New message registration: {"p":"", "o":"add", "v": {"message":...}}
        if o == "add" and p == "" and isinstance(v, dict) and "message" in v:
            register(v)
            last_path_op = None
            continue

        # Shorthand add (no p/o, v has message+conversation_id).
        if p is None and o is None and isinstance(v, dict) and "message" in v:
            register(v)
            last_path_op = None
            continue

        # Batched patch: {"o":"patch", "v":[ops...]}.
        if o == "patch" and isinstance(v, list):
            for op in v:
                if isinstance(op, dict):
                    apply_op(op.get("p"), op.get("o"), op.get("v"))
            last_path_op = None
            continue

        # Shorthand batched patch: {"v":[ops...]}.
        if p is None and o is None and isinstance(v, list):
            for op in v:
                if isinstance(op, dict):
                    apply_op(op.get("p"), op.get("o"), op.get("v"))
            last_path_op = None
            continue

        # Direct single op with explicit path & op.
        if isinstance(p, str) and isinstance(o, str):
            apply_op(p, o, v)
            continue

        # Shorthand continuation: {"v": "<more text>"} reuses last (p, o).
        if p is None and o is None and last_path_op is not None and isinstance(v, str):
            lp, lo = last_path_op
            apply_op(lp, lo, v)
            continue

    # Visible assistant answer: role=assistant, recipient=all, content_type=text.
    pieces = [
        m["parts0"]
        for m in messages.values()
        if m.get("role") == "assistant"
        and m.get("recipient") == "all"
        and m.get("content_type") == "text"
        and isinstance(m.get("parts0"), str)
        and m["parts0"].strip()
    ]
    return "\n\n".join(pieces).strip()


# ---------------------------------------------------------------------------
# main extraction


CONV_URL_SUFFIX = "/backend-api/f/conversation"


def extract(har_path: str):
    with open(har_path, "r", encoding="utf-8") as f:
        har = json.load(f)

    entries = ((har.get("log") or {}).get("entries")) or []
    turns = []

    for entry in entries:
        req = entry.get("request") or {}
        url = req.get("url") or ""
        if req.get("method") != "POST":
            continue
        # Match the conversation endpoint (ignore /prepare etc.)
        if not url.endswith(CONV_URL_SUFFIX):
            continue

        req_text = get_body_text(req)
        if not req_text:
            continue
        try:
            req_obj = json.loads(req_text)
        except Exception:
            continue
        if req_obj.get("action") != "next":
            continue
        msgs = req_obj.get("messages") or []
        if not msgs or not isinstance(msgs[0], dict):
            continue

        m0 = msgs[0]
        parts = ((m0.get("content") or {}).get("parts")) or []
        user_text = parts[0] if parts else ""
        if not isinstance(user_text, str) or not user_text.strip():
            continue

        entry_start = parse_iso8601(entry.get("startedDateTime") or "")
        ct = m0.get("create_time")
        user_ts = int(ct) if isinstance(ct, (int, float)) and ct > 0 else entry_start

        # Reconstruct assistant response.
        resp_text = get_body_text(entry.get("response") or {})
        asst_text = reconstruct_assistant(resp_text) if resp_text else ""

        dur_ms = entry.get("time") or 0
        try:
            dur_s = int(float(dur_ms) / 1000.0)
        except Exception:
            dur_s = 0
        asst_ts = (entry_start or user_ts) + max(dur_s, 1)

        turns.append({
            "role": "user",
            "content": user_text,
            "endpoint": url,
            "timestamp": user_ts,
        })
        if asst_text:
            turns.append({
                "role": "assistant",
                "content": asst_text,
                "endpoint": url,
                "timestamp": asst_ts,
            })

    # Enforce strictly monotonic timestamps while preserving HAR order.
    prev = 0
    for t in turns:
        if t["timestamp"] <= prev:
            t["timestamp"] = prev + 1
        prev = t["timestamp"]

    out = []
    for i, t in enumerate(turns):
        out.append({
            "turn": i,
            "role": t["role"],
            "content": t["content"],
            "endpoint": t["endpoint"],
            "timestamp": t["timestamp"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    turns = extract(args.har)
    Path(args.out).write_text(
        json.dumps(turns, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
