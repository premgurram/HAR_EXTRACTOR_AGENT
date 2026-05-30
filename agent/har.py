"""Provider-agnostic HAR helpers.

Knows the HAR 1.2 spec only. No knowledge of any specific chat provider,
URL, or JSON shape lives here.
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Entry:
    index: int
    method: str
    url: str
    status: int
    mime: str
    started_at: str
    request_headers: list[dict[str, str]]
    response_headers: list[dict[str, str]]
    request_body: bytes
    response_body: bytes

    @property
    def req_bytes(self) -> int:
        return len(self.request_body)

    @property
    def resp_bytes(self) -> int:
        return len(self.response_body)


def load_har(path: str | Path) -> list[Entry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries_raw = raw.get("log", {}).get("entries", [])
    out: list[Entry] = []
    for i, e in enumerate(entries_raw):
        req = e.get("request", {}) or {}
        resp = e.get("response", {}) or {}
        content = resp.get("content", {}) or {}
        out.append(
            Entry(
                index=i,
                method=req.get("method", ""),
                url=req.get("url", ""),
                status=resp.get("status", 0) or 0,
                mime=content.get("mimeType", "") or "",
                started_at=e.get("startedDateTime", "") or "",
                request_headers=list(req.get("headers", []) or []),
                response_headers=list(resp.get("headers", []) or []),
                request_body=_decode_request_body(req),
                response_body=_decode_response_body(resp),
            )
        )
    return out


def _decode_request_body(req: dict[str, Any]) -> bytes:
    pd = req.get("postData") or {}
    text = pd.get("text")
    if text is None:
        return b""
    if isinstance(text, bytes):
        return text
    # postData.text is normally already a string; HAR has no encoding field here.
    return text.encode("utf-8", errors="replace")


def _decode_response_body(resp: dict[str, Any]) -> bytes:
    content = resp.get("content") or {}
    text = content.get("text")
    if text is None:
        return b""
    encoding = (content.get("encoding") or "").lower()
    raw: bytes
    if encoding == "base64":
        try:
            raw = base64.b64decode(text)
        except Exception:
            return text.encode("utf-8", errors="replace")
    else:
        raw = text.encode("utf-8", errors="replace") if isinstance(text, str) else text

    # HAR captures usually contain already-decompressed bodies, but some tools
    # (mitmproxy, charles) preserve the wire bytes. Try to decompress if a
    # content-encoding header suggests it.
    enc = _header(resp.get("headers", []), "content-encoding").lower()
    if "gzip" in enc:
        try:
            return gzip.decompress(raw)
        except Exception:
            pass
    if "deflate" in enc:
        try:
            return zlib.decompress(raw)
        except Exception:
            try:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
            except Exception:
                pass
    if "br" in enc:
        try:
            import brotli  # type: ignore

            return brotli.decompress(raw)
        except Exception:
            pass
    return raw


def _header(headers: Iterable[dict[str, str]], name: str) -> str:
    name_l = name.lower()
    for h in headers:
        if (h.get("name") or "").lower() == name_l:
            return h.get("value") or ""
    return ""


def body_as_text(body: bytes, limit: int | None = None) -> str:
    try:
        s = body.decode("utf-8", errors="replace")
    except Exception:
        s = repr(body)
    if limit is not None and len(s) > limit:
        return s[:limit] + f"\n...[truncated {len(s) - limit} bytes]"
    return s


def looks_like_streaming(mime: str, body: bytes) -> bool:
    m = (mime or "").lower()
    if "event-stream" in m or "ndjson" in m or "x-ndjson" in m:
        return True
    head = body[:512]
    if head.startswith(b"data:") or b"\ndata:" in head:
        return True
    # NDJSON heuristic: multiple newline-separated JSON objects
    lines = [ln for ln in body.splitlines()[:5] if ln.strip()]
    if len(lines) >= 2 and all(ln.startswith(b"{") and ln.endswith(b"}") for ln in lines):
        return True
    return False
