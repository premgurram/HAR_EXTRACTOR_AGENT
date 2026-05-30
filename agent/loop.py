from __future__ import annotations

import json
from typing import Any

from agent import tools
from agent.llm import LLMProvider
from agent.prompts import SYSTEM_PROMPT
from agent.tracer import Tracer, preview


DEFAULT_MAX_TURNS = 25
DEFAULT_MAX_TOKENS = 8192


def run_agent(
    *,
    har_path: str,
    extractor_out_path: str,
    tracer: Tracer,
    provider: LLMProvider,
    model: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    ctx = tools.make_context(har_path, extractor_out_path)
    chosen_model = model or provider.default_model

    tracer.event(
        "run_start",
        har_path=har_path,
        extractor_out_path=extractor_out_path,
        har_entry_count=len(ctx.entries),
        llm_provider=provider.name,
        model=chosen_model,
        max_turns=max_turns,
    )
    tracer.event("system_prompt", text=SYSTEM_PROMPT)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"HAR file loaded: {har_path}\n"
                f"Entries in HAR: {len(ctx.entries)}\n"
                f"Extractor output path: {extractor_out_path}\n\n"
                "Begin with `list_entries` (sort_by=\"score\" is the default)."
            ),
        }
    ]

    turns_used = 0
    last_text = ""
    for turn in range(max_turns):
        turns_used = turn + 1
        tracer.event("model_request", turn=turn, messages_count=len(messages))

        response = provider.chat(
            system=SYSTEM_PROMPT,
            tools=tools.TOOL_DEFS,
            history=messages,
            model=chosen_model,
            max_tokens=max_tokens,
        )

        tracer.event(
            "model_response",
            turn=turn,
            stop_reason=response.get("stop_reason"),
            **{k: v for k, v in (response.get("usage") or {}).items()},
        )

        assistant_content: list[dict[str, Any]] = []
        tool_uses: list[dict[str, Any]] = []
        for block in response.get("blocks", []):
            btype = block.get("type")
            if btype == "thinking":
                assistant_content.append(block)
                tracer.event(
                    "model_thinking",
                    turn=turn,
                    text=preview(block.get("thinking", ""), 1500),
                )
            elif btype == "text":
                assistant_content.append(block)
                last_text = block.get("text", "")
                tracer.event("model_text", turn=turn, text=preview(last_text, 1500))
            elif btype == "tool_use":
                assistant_content.append(block)
                tool_uses.append(block)
                tracer.event(
                    "tool_call",
                    turn=turn,
                    tool=block.get("name"),
                    input=preview(block.get("input") or {}, 600),
                    id=block.get("id"),
                )

        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_uses:
            tracer.event(
                "model_end_turn_no_tools",
                turn=turn,
                stop_reason=response.get("stop_reason"),
            )
            break

        tool_results: list[dict[str, Any]] = []
        for tu in tool_uses:
            result = tools.dispatch(ctx, tu.get("name", ""), tu.get("input") or {})
            tracer.event(
                "tool_result",
                turn=turn,
                tool=tu.get("name"),
                id=tu.get("id"),
                result=preview(result, 1500),
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.get("id", ""),
                "content": _serialize_for_wire(result),
            })

        messages.append({"role": "user", "content": tool_results})

        if ctx.finalized:
            tracer.event("finalize_observed", turn=turn, note=ctx.finalize_note)
            break

    outcome = {
        "finalized": ctx.finalized,
        "finalize_note": ctx.finalize_note,
        "turns_used": turns_used,
        "extractor_path": str(ctx.extractor_out_path) if ctx.extractor_out_path.exists() else None,
        "last_model_text": preview(last_text, 600),
        "llm_provider": provider.name,
        "model": chosen_model,
    }
    tracer.event("run_end", **outcome)
    return outcome


def _serialize_for_wire(result: Any) -> str:
    try:
        return json.dumps(result, default=str)
    except Exception:
        return repr(result)
