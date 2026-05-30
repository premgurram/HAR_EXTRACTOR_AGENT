from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any


# ----------------------------------------------------------------------------
# Unified internal shapes (Anthropic-style content blocks)
#
# History is a list of {role: "user"|"assistant", content: list[block] | str}.
# Block types:
#   {"type": "text",        "text": str}
#   {"type": "tool_use",    "id": str, "name": str, "input": dict}
#   {"type": "tool_result", "tool_use_id": str, "content": str}
#   {"type": "thinking",    "thinking": str, "signature": str}    (Anthropic only)
#
# Tools are Anthropic-shaped: {name, description, input_schema}.
#
# chat() returns: {
#   "blocks":      list[block]            # text / tool_use / thinking
#   "stop_reason": "end_turn"|"tool_use"|"max_tokens"|str
#   "usage":       dict[str, int | None]  # provider-specific keys
# }
# ----------------------------------------------------------------------------


class LLMProvider(ABC):
    name: str = ""
    default_model: str = ""

    @abstractmethod
    def chat(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],
        history: list[dict[str, Any]],
        model: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        ...


# ---- Anthropic --------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    name = "claude"
    default_model = "claude-opus-4-7"

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def chat(self, *, system, tools, history, model, max_tokens):
        cached_history = _with_message_cache_breakpoint(history)
        cached_tools = [
            {**t, "cache_control": {"type": "ephemeral"}} if i == len(tools) - 1 else t
            for i, t in enumerate(tools)
        ]
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive", "display": "summarized"},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=cached_tools,
            messages=cached_history,
        )
        blocks: list[dict[str, Any]] = []
        for b in response.content:
            if b.type == "thinking":
                blocks.append({
                    "type": "thinking",
                    "thinking": getattr(b, "thinking", "") or "",
                    "signature": getattr(b, "signature", "") or "",
                })
            elif b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({
                    "type": "tool_use",
                    "id": b.id,
                    "name": b.name,
                    "input": dict(b.input) if b.input else {},
                })
        return {
            "blocks": blocks,
            "stop_reason": _norm_anthropic_stop(response.stop_reason),
            "usage": {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", None),
            },
        }


# ---- OpenAI / OpenAI-compatible --------------------------------------------


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions provider.

    Subclasses override `name`, `default_model`, `env_var`, and `base_url`
    for other OpenAI-compatible vendors (Kimi, etc).
    """

    name = "openai"
    default_model = "gpt-4o"
    env_var = "OPENAI_API_KEY"
    base_url: str | None = None

    def __init__(self, api_key: str | None = None) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or os.environ.get(self.env_var),
            base_url=self.base_url,
        )

    def chat(self, *, system, tools, history, model, max_tokens):
        o_messages = _to_openai_messages(system, history)
        o_tools = _to_openai_tools(tools)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": o_messages,
        }
        if o_tools:
            kwargs["tools"] = o_tools
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        blocks: list[dict[str, Any]] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": args,
            })
        usage_obj = getattr(response, "usage", None)
        return {
            "blocks": blocks,
            "stop_reason": _norm_openai_stop(choice.finish_reason),
            "usage": {
                "input_tokens": getattr(usage_obj, "prompt_tokens", None),
                "output_tokens": getattr(usage_obj, "completion_tokens", None),
            },
        }


class KimiProvider(OpenAIProvider):
    """Moonshot AI's Kimi via the OpenAI-compatible Chat Completions endpoint."""

    name = "kimi"
    default_model = "kimi-k2-0905-preview"
    env_var = "MOONSHOT_API_KEY"
    base_url = "https://api.moonshot.ai/v1"


# ---- Factory ---------------------------------------------------------------


def make_provider(name: str) -> LLMProvider:
    n = (name or "").lower()
    if n in ("claude", "anthropic"):
        return AnthropicProvider()
    if n in ("openai", "gpt"):
        return OpenAIProvider()
    if n in ("kimi", "moonshot"):
        return KimiProvider()
    raise ValueError(f"unknown LLM provider: {name!r}; expected one of: claude, openai, kimi")


# ---- Anthropic-specific cache placement ------------------------------------


def _with_message_cache_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add a cache_control breakpoint to the last block of the most-recent user
    message, so the growing conversation prefix caches incrementally.
    """
    if not messages:
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            msg = dict(out[i])
            content = msg.get("content")
            if isinstance(content, list) and content:
                new_content = list(content)
                last = new_content[-1]
                if isinstance(last, dict):
                    last = dict(last)
                    last["cache_control"] = {"type": "ephemeral"}
                    new_content[-1] = last
                    msg["content"] = new_content
                    out[i] = msg
            elif isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
                out[i] = msg
            break
    return out


# ---- OpenAI conversion -----------------------------------------------------


def _to_openai_messages(system: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        if role == "user":
            text_parts: list[str] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    text_parts.append(b.get("text", ""))
                elif t == "tool_result":
                    if text_parts:
                        msgs.append({"role": "user", "content": "\n".join(text_parts)})
                        text_parts = []
                    tc = b.get("content")
                    if not isinstance(tc, str):
                        tc = json.dumps(tc, default=str)
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id", ""),
                        "content": tc,
                    })
            if text_parts:
                msgs.append({"role": "user", "content": "\n".join(text_parts)})
        else:  # assistant
            text_parts2: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    text_parts2.append(b.get("text", ""))
                elif t == "tool_use":
                    tool_calls.append({
                        "id": b.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input") or {}),
                        },
                    })
                # thinking blocks: skipped for OpenAI/Kimi
            asst: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts2) if text_parts2 else None,
            }
            if tool_calls:
                asst["tool_calls"] = tool_calls
            msgs.append(asst)
    return msgs


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


# ---- Stop-reason normalization ---------------------------------------------


def _norm_anthropic_stop(s: str | None) -> str:
    if s in ("tool_use", "end_turn", "max_tokens"):
        return s
    return s or "end_turn"


def _norm_openai_stop(s: str | None) -> str:
    if s == "tool_calls":
        return "tool_use"
    if s == "stop":
        return "end_turn"
    if s == "length":
        return "max_tokens"
    return s or "end_turn"
