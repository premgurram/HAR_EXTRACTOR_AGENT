"""Append-only JSON trace logger for agent runs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict[str, Any]] = []
        self._t0 = time.time()

    def event(self, kind: str, **fields: Any) -> None:
        ev = {"t": round(time.time() - self._t0, 3), "kind": kind, **fields}
        self._events.append(ev)
        # Flush after every event so partial runs are inspectable.
        self.path.write_text(
            json.dumps(self._events, indent=2, default=_safe_default),
            encoding="utf-8",
        )


def _safe_default(o: Any) -> Any:
    if isinstance(o, bytes):
        try:
            return o.decode("utf-8", errors="replace")
        except Exception:
            return repr(o)
    return repr(o)


def preview(value: Any, limit: int = 800) -> Any:
    """Truncate long strings/objects for trace storage."""
    if isinstance(value, str):
        if len(value) > limit:
            return value[:limit] + f"...[+{len(value) - limit} chars]"
        return value
    if isinstance(value, (list, tuple)):
        return [preview(v, limit) for v in value[:20]]
    if isinstance(value, dict):
        return {k: preview(v, limit) for k, v in list(value.items())[:30]}
    return value
