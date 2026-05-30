"""Tool schemas + implementations exposed to the agent loop.

Each tool returns a dict that's safe to embed in an Anthropic tool_result
content block (plain JSON). The dict shapes match the schemas declared in
TOOL_DEFS below.

These tools know nothing about specific chat providers. They speak HAR.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent import har, sandbox, verify
from agent.tracer import preview


# JSON-schema definitions for Anthropic's tool-use format.
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "list_entries",
        "description": (
            "Return a compact index of HAR entries with a conversational-likelihood "
            "score per entry. Score signals: POST method, JSON / event-stream / "
            "NDJSON response mime, reasonable response size, non-trivial request "
            "body. Each entry includes `score` (int, higher = more likely chat) "
            "and `score_reasons` (list of human-readable hints). Set "
            "`sort_by=\"score\"` (default) to see best candidates first; "
            "`sort_by=\"index\"` preserves capture order. `drop_static` (default "
            "true) omits images/css/fonts/preflights. Returns at most `limit` "
            "entries (default 50)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url_filter": {"type": "string"},
                "drop_static": {"type": "boolean", "default": True},
                "sort_by": {"type": "string", "enum": ["score", "index", "size"], "default": "score"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "inspect_entry",
        "description": (
            "Return headers and a body slice for one HAR entry. `body` selects "
            "'request' or 'response'. `max_bytes` caps the returned body text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "body": {"type": "string", "enum": ["request", "response", "both"], "default": "both"},
                "max_bytes": {"type": "integer", "default": 4096},
            },
            "required": ["index"],
        },
    },
    {
        "name": "search_bodies",
        "description": (
            "Substring search across HAR bodies. `where` selects 'request', "
            "'response', or 'both'. Returns hits with entry index, URL, byte "
            "offset, and a short snippet around the match. Case-sensitive."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "where": {"type": "string", "enum": ["request", "response", "both"], "default": "both"},
                "max_hits": {"type": "integer", "default": 20},
                "snippet_chars": {"type": "integer", "default": 200},
            },
            "required": ["query"],
        },
    },
    {
        "name": "decode_stream",
        "description": (
            "Best-effort parser for a streamed response body. Handles SSE "
            "('data: ...\\n\\n'), NDJSON, and concatenated JSON. Returns the "
            "reconstructed full text (if discoverable from common text fields), "
            "the chunk format, a list of distinct JSON key paths seen, and a "
            "sample of the first few chunks for the agent to inspect."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "max_chunks_sample": {"type": "integer", "default": 5},
            },
            "required": ["index"],
        },
    },
    {
        "name": "sample_body_slices",
        "description": (
            "Return head, middle, and tail slices of a response (or request) body. "
            "Use this when a body is large (typical of streaming responses) and a "
            "single head-only inspection would miss the actual content chunks. "
            "`slice_bytes` controls the per-slice size (default 1500)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "body": {"type": "string", "enum": ["request", "response"], "default": "response"},
                "slice_bytes": {"type": "integer", "default": 1500},
            },
            "required": ["index"],
        },
    },
    {
        "name": "write_extractor",
        "description": (
            "Write the standalone extractor script. The file will be written to "
            "the path supplied at agent launch; only the source code is needed here. "
            "Subsequent calls overwrite the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "run_extractor",
        "description": (
            "Run the last-written extractor in a sandbox on the HAR. Returns "
            "the parsed JSON output (or stderr/error)."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "verify_structure",
        "description": (
            "Check the last extractor run's output against structural rules: "
            "turns indexed 0..N-1, roles alternate user/assistant, content "
            "non-empty, timestamps monotonic and epoch-shaped, endpoints "
            "valid http(s) URLs, at least 2 turns. Returns ok, "
            "structural_issues, summary. Does NOT compare against any "
            "ground truth - that is done outside this loop. Call after "
            "run_extractor."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "finalize",
        "description": (
            "Mark the run complete. Only call after verify_structure returned ok=true. "
            "Provide a brief note describing your final extractor's approach."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    },
]


_STATIC_MIME_HINTS = ("image/", "font/", "text/css", "text/html", "video/", "audio/")
_STATIC_PATH_HINTS = re.compile(r"\.(png|jpg|jpeg|gif|webp|svg|ico|woff2?|ttf|css|js|map|mp4|webm)(\?|$)", re.IGNORECASE)


@dataclass
class ToolContext:
    har_path: Path
    extractor_out_path: Path
    entries: list[har.Entry]
    last_run: sandbox.RunResult | None = None
    finalized: bool = False
    finalize_note: str | None = None


def make_context(har_path: str | Path, extractor_out_path: str | Path) -> ToolContext:
    entries = har.load_har(har_path)
    return ToolContext(
        har_path=Path(har_path),
        extractor_out_path=Path(extractor_out_path),
        entries=entries,
    )


def dispatch(ctx: ToolContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(ctx, args)
    except Exception as e:  # pragma: no cover - surfaced to the agent
        return {"error": f"{type(e).__name__}: {e}"}


# ---- individual tools ------------------------------------------------------


def _tool_list_entries(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    url_filter = (args.get("url_filter") or "").lower()
    drop_static = args.get("drop_static", True)
    sort_by = args.get("sort_by", "score")
    limit = int(args.get("limit", 50))
    scored: list[dict[str, Any]] = []
    for e in ctx.entries:
        if drop_static and _is_static(e):
            continue
        if url_filter and url_filter not in e.url.lower():
            continue
        score, reasons = _conversational_score(e)
        scored.append({
            "i": e.index,
            "method": e.method,
            "url": e.url,
            "status": e.status,
            "mime": e.mime,
            "started_at": e.started_at,
            "req_bytes": e.req_bytes,
            "resp_bytes": e.resp_bytes,
            "score": score,
            "score_reasons": reasons,
        })
    if sort_by == "score":
        scored.sort(key=lambda x: (-x["score"], -x["resp_bytes"]))
    elif sort_by == "size":
        scored.sort(key=lambda x: -x["resp_bytes"])
    # else "index": preserve capture order (already in that order)
    out = scored[:limit]
    return {"count": len(out), "total_in_har": len(ctx.entries), "entries": out}


def _conversational_score(e: har.Entry) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if e.method == "POST":
        score += 2
        reasons.append("POST method (+2)")
    mime = (e.mime or "").lower()
    if "event-stream" in mime:
        score += 3
        reasons.append("event-stream response (+3)")
    elif "ndjson" in mime or "x-ndjson" in mime:
        score += 3
        reasons.append("NDJSON response (+3)")
    elif "json" in mime:
        score += 2
        reasons.append("JSON response (+2)")
    # Body looks like SSE/NDJSON even if mime is wrong
    head = e.response_body[:256]
    if score < 3 and (head.startswith(b"data:") or b"\ndata:" in head):
        score += 2
        reasons.append("body looks like SSE (+2)")
    if 500 <= e.resp_bytes <= 5_000_000:
        score += 1
        reasons.append(f"reasonable response size ({e.resp_bytes}B, +1)")
    elif e.resp_bytes > 5_000_000:
        score -= 1
        reasons.append(f"response very large ({e.resp_bytes}B, -1)")
    if e.req_bytes > 30 and e.method in ("POST", "PUT", "PATCH"):
        score += 1
        reasons.append(f"non-trivial request body ({e.req_bytes}B, +1)")
    # Penalize URLs that look like telemetry
    url_l = e.url.lower()
    for needle in ("telemetry", "/log", "ping", "beacon", "analytics", "metrics", "sentry"):
        if needle in url_l:
            score -= 2
            reasons.append(f"telemetry-looking URL ({needle}, -2)")
            break
    return score, reasons


def _is_static(e: har.Entry) -> bool:
    m = (e.mime or "").lower()
    if any(m.startswith(h) for h in _STATIC_MIME_HINTS):
        # Keep text/html only when it might be an SSE response under a misreported mime
        if m.startswith("text/html") and e.resp_bytes > 5000:
            return False
        return True
    if _STATIC_PATH_HINTS.search(e.url):
        return True
    # Drop OPTIONS preflights and obviously-empty entries
    if e.method == "OPTIONS":
        return True
    if e.resp_bytes == 0 and e.req_bytes < 16:
        return True
    return False


def _tool_inspect_entry(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    i = int(args["index"])
    body = args.get("body", "both")
    max_bytes = int(args.get("max_bytes", 4096))
    if i < 0 or i >= len(ctx.entries):
        return {"error": f"index {i} out of range (0..{len(ctx.entries)-1})"}
    e = ctx.entries[i]
    result: dict[str, Any] = {
        "i": e.index,
        "method": e.method,
        "url": e.url,
        "status": e.status,
        "mime": e.mime,
        "started_at": e.started_at,
        "request_headers": _h_summary(e.request_headers),
        "response_headers": _h_summary(e.response_headers),
        "req_bytes": e.req_bytes,
        "resp_bytes": e.resp_bytes,
    }
    if body in ("request", "both"):
        result["request_body"] = har.body_as_text(e.request_body, max_bytes)
    if body in ("response", "both"):
        result["response_body"] = har.body_as_text(e.response_body, max_bytes)
        result["response_looks_streaming"] = har.looks_like_streaming(e.mime, e.response_body)
    return result


def _h_summary(headers: list[dict[str, str]]) -> list[dict[str, str]]:
    keep = {"content-type", "content-encoding", "transfer-encoding", "accept", "x-requested-with"}
    out = []
    for h in headers:
        name = (h.get("name") or "").lower()
        if name in keep or name.startswith(":"):
            out.append({"name": h.get("name", ""), "value": h.get("value", "")})
    return out


def _tool_search_bodies(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    q = args["query"]
    if not isinstance(q, str) or not q:
        return {"error": "query must be a non-empty string"}
    where = args.get("where", "both")
    max_hits = int(args.get("max_hits", 20))
    snippet_chars = int(args.get("snippet_chars", 200))
    qb = q.encode("utf-8", errors="replace")

    hits: list[dict[str, Any]] = []
    for e in ctx.entries:
        for which, body in (("request", e.request_body), ("response", e.response_body)):
            if where != "both" and which != where:
                continue
            if not body:
                continue
            off = body.find(qb)
            if off < 0:
                continue
            start = max(0, off - snippet_chars // 2)
            end = min(len(body), off + len(qb) + snippet_chars // 2)
            hits.append({
                "i": e.index,
                "url": e.url,
                "where": which,
                "offset": off,
                "mime": e.mime,
                "snippet": har.body_as_text(body[start:end]),
            })
            if len(hits) >= max_hits:
                return {"hits": hits, "truncated": True}
    return {"hits": hits, "truncated": False}


def _tool_decode_stream(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    i = int(args["index"])
    max_sample = int(args.get("max_chunks_sample", 5))
    if i < 0 or i >= len(ctx.entries):
        return {"error": f"index {i} out of range"}
    e = ctx.entries[i]
    body = e.response_body
    if not body:
        return {"error": "response body is empty"}

    text = body.decode("utf-8", errors="replace")
    result: dict[str, Any] = {"i": e.index, "url": e.url, "mime": e.mime}

    # Try SSE
    sse_events = _parse_sse(text)
    if sse_events:
        result["format"] = "sse"
        result["event_count"] = len(sse_events)
        result["sample_events"] = sse_events[:max_sample]
        reconstructed, key_paths = _reconstruct_from_json_events(sse_events)
        result["json_key_paths_seen"] = key_paths
        if reconstructed is not None:
            result["reconstructed_text_preview"] = preview(reconstructed, 1200)
            result["reconstructed_text_length"] = len(reconstructed)
        return result

    # Try NDJSON
    nd_events = _parse_ndjson(text)
    if nd_events:
        result["format"] = "ndjson"
        result["event_count"] = len(nd_events)
        result["sample_events"] = nd_events[:max_sample]
        reconstructed, key_paths = _reconstruct_from_json_events(nd_events)
        result["json_key_paths_seen"] = key_paths
        if reconstructed is not None:
            result["reconstructed_text_preview"] = preview(reconstructed, 1200)
            result["reconstructed_text_length"] = len(reconstructed)
        return result

    # Try single JSON
    try:
        obj = json.loads(text)
        result["format"] = "json"
        result["object_preview"] = preview(obj, 1200)
        result["json_key_paths_seen"] = sorted(_walk_paths(obj))[:80]
        return result
    except json.JSONDecodeError:
        pass

    result["format"] = "unknown"
    result["body_preview"] = preview(text, 1200)
    return result


def _parse_sse(text: str) -> list[dict[str, Any]] | None:
    if "\ndata:" not in text and not text.lstrip().startswith("data:"):
        return None
    events: list[dict[str, Any]] = []
    current_data: list[str] = []
    current_event: str | None = None
    for line in text.splitlines():
        if line == "":
            if current_data:
                joined = "\n".join(current_data)
                parsed: Any
                try:
                    parsed = json.loads(joined)
                except json.JSONDecodeError:
                    parsed = joined
                events.append({"event": current_event, "data": parsed})
            current_data = []
            current_event = None
            continue
        if line.startswith("data:"):
            current_data.append(line[5:].lstrip())
        elif line.startswith("event:"):
            current_event = line[6:].strip()
    if current_data:
        joined = "\n".join(current_data)
        try:
            parsed = json.loads(joined)
        except json.JSONDecodeError:
            parsed = joined
        events.append({"event": current_event, "data": parsed})
    return events or None


def _parse_ndjson(text: str) -> list[Any] | None:
    out: list[Any] = []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            return None
    return out


def _walk_paths(obj: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            paths.add(p)
            paths |= _walk_paths(v, p)
    elif isinstance(obj, list):
        for v in obj[:5]:  # only first few children
            paths |= _walk_paths(v, prefix + "[]")
    return paths


def _reconstruct_from_json_events(events: list[Any]) -> tuple[str | None, list[str]]:
    """Try to concatenate text from a stream of JSON events.

    Looks for any string-valued field whose name suggests text content and
    accumulates them. We don't hardcode a single key name; we collect every
    leaf string field and return the one that produces the longest coherent
    text. The agent will see the key paths and can mirror them precisely.
    """
    by_path: dict[str, list[str]] = {}
    seen_paths: set[str] = set()

    for ev in events:
        data = ev.get("data") if isinstance(ev, dict) and "data" in ev else ev
        if isinstance(data, str):
            by_path.setdefault("<raw_string>", []).append(data)
            seen_paths.add("<raw_string>")
            continue
        if not isinstance(data, (dict, list)):
            continue
        for path, value in _iter_string_leaves(data):
            seen_paths.add(path)
            by_path.setdefault(path, []).append(value)

    if not by_path:
        return None, sorted(seen_paths)

    best_path = max(by_path.keys(), key=lambda p: sum(len(s) for s in by_path[p]))
    return "".join(by_path[best_path]), sorted(seen_paths)


def _iter_string_leaves(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.extend(_iter_string_leaves(v, p))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_iter_string_leaves(v, prefix + "[]"))
    elif isinstance(obj, str):
        out.append((prefix or "<root>", obj))
    return out


def _tool_sample_body_slices(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    i = int(args["index"])
    which = args.get("body", "response")
    slice_bytes = int(args.get("slice_bytes", 1500))
    if i < 0 or i >= len(ctx.entries):
        return {"error": f"index {i} out of range"}
    e = ctx.entries[i]
    body = e.response_body if which == "response" else e.request_body
    n = len(body)
    if n == 0:
        return {"error": f"{which} body is empty"}
    if n <= slice_bytes * 3:
        # Small enough to return whole
        return {
            "i": e.index,
            "url": e.url,
            "which": which,
            "total_bytes": n,
            "complete": True,
            "body": har.body_as_text(body),
        }
    head = body[:slice_bytes]
    mid_start = max(slice_bytes, n // 2 - slice_bytes // 2)
    mid = body[mid_start : mid_start + slice_bytes]
    tail = body[-slice_bytes:]
    return {
        "i": e.index,
        "url": e.url,
        "which": which,
        "total_bytes": n,
        "complete": False,
        "head": har.body_as_text(head),
        "head_offset": 0,
        "middle": har.body_as_text(mid),
        "middle_offset": mid_start,
        "tail": har.body_as_text(tail),
        "tail_offset": n - len(tail),
    }


def _tool_write_extractor(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return {"error": "code must be a non-empty string"}
    lint = sandbox.lint_extractor_source(code)
    if not lint.ok:
        return {"ok": False, "wrote": False, "lint_issues": lint.issues}
    ctx.extractor_out_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.extractor_out_path.write_text(code, encoding="utf-8")
    return {"ok": True, "wrote": True, "path": str(ctx.extractor_out_path), "bytes": len(code)}


def _tool_run_extractor(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if not ctx.extractor_out_path.exists():
        return {"error": "no extractor written yet; call write_extractor first"}
    result = sandbox.run_extractor(ctx.extractor_out_path, ctx.har_path)
    ctx.last_run = result
    payload: dict[str, Any] = {
        "ok": result.ok,
        "stdout": preview(result.stdout, 400),
        "stderr": preview(result.stderr, 1200),
        "error": result.error,
    }
    if result.ok and result.output_json is not None:
        # Don't dump the whole log; the verify tool gives precise feedback.
        if isinstance(result.output_json, list):
            payload["output_summary"] = {
                "turns": len(result.output_json),
                "first_turn": preview(result.output_json[0] if result.output_json else None, 400),
                "last_turn": preview(result.output_json[-1] if result.output_json else None, 400),
            }
        else:
            payload["output_preview"] = preview(result.output_json, 600)
    return payload


def _tool_verify_structure(ctx: ToolContext, _args: dict[str, Any]) -> dict[str, Any]:
    if ctx.last_run is None or not ctx.last_run.ok or ctx.last_run.output_json is None:
        return {"error": "no successful extractor run yet; call run_extractor first"}
    if not isinstance(ctx.last_run.output_json, list):
        return {"error": "extractor output is not a list", "got_type": type(ctx.last_run.output_json).__name__}
    # Structural check only - no ground truth. The post-run CLI does that
    # comparison separately for the human reviewer.
    result = verify.verify(ctx.last_run.output_json, ground_truth_prompts=None)
    clean_summary = {k: v for k, v in result.summary.items() if not k.startswith("ground_truth")}
    return {
        "ok": result.ok,
        "structural_issues": result.structural_issues,
        "summary": clean_summary,
    }


def _tool_finalize(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    note = args.get("note") or ""
    if ctx.last_run is None or ctx.last_run.output_json is None:
        return {"error": "cannot finalize: no extractor output yet"}
    if not isinstance(ctx.last_run.output_json, list):
        return {"error": "cannot finalize: last output is not a list"}
    v = verify.verify(ctx.last_run.output_json, ground_truth_prompts=None)
    if not v.ok:
        return {"error": "cannot finalize: verify_structure still not ok", "details": v.to_dict()}
    ctx.finalized = True
    ctx.finalize_note = note
    return {"ok": True, "note": note, "summary": v.summary}


_DISPATCH = {
    "list_entries": _tool_list_entries,
    "inspect_entry": _tool_inspect_entry,
    "search_bodies": _tool_search_bodies,
    "decode_stream": _tool_decode_stream,
    "sample_body_slices": _tool_sample_body_slices,
    "write_extractor": _tool_write_extractor,
    "run_extractor": _tool_run_extractor,
    "verify_structure": _tool_verify_structure,
    "finalize": _tool_finalize,
}
