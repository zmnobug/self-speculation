"""Qwen prompt helpers for the paper-aligned tagged-JSON tool protocol."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


QWEN_JSON_TOOL_PROTOCOL = """Use the tools described below when needed.

Available tools (OpenAI JSON schema):
{tools_json}

When calling a tool, emit exactly one call in this format:
<tool_call>
{{"name":"tool_name","arguments":{{"argument_name":"value"}}}}
</tool_call>

Do not emit XML <function> or <parameter> tags. Call at most one tool per turn."""


def is_qwen35_family(model_name: str, model_family: str = "auto") -> bool:
    """Recognize Qwen3.5 through Qwen3.9 deployment and class aliases."""

    if model_family != "auto":
        if model_family not in {"qwen3", "qwen3.5"}:
            raise ValueError(f"unknown Qwen model family: {model_family}")
        return model_family == "qwen3.5"
    # An underscore is used as a class-name separator (for example
    # ``Qwen3_5ForConditionalGeneration``), while a hyphen commonly separates
    # the base family from its parameter count (``Qwen3-8B``).  Treating both
    # as decimal separators would misclassify ordinary Qwen3-8B deployments as
    # Qwen3.8.
    normalized = model_name.lower().replace("_", ".")
    aliases = (
        "qwen35",
        "qwen36",
        "qwen37",
        "qwen38",
        "qwen39",
        "qwen3.5for",
    )
    return bool(re.search(r"qwen3\.[5-9]", normalized)) or any(
        alias in normalized for alias in aliases
    )


def with_qwen_json_tool_protocol(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Copy messages and place JSON tool instructions in the system message.

    Qwen3.5-family native templates may switch to XML when passed ``tools=``.
    Injecting the schemas as text lets callers intentionally reproduce SPORK's
    tagged-JSON protocol without mutating the original conversation.
    """

    rendered = [dict(message) for message in messages]
    if not tools:
        return tuple(rendered)
    protocol = QWEN_JSON_TOOL_PROTOCOL.format(
        tools_json=json.dumps(
            list(tools),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    for message in rendered:
        if message.get("role") == "system":
            content = str(message.get("content") or "").rstrip()
            message["content"] = f"{content}\n\n{protocol}" if content else protocol
            break
    else:
        rendered.insert(0, {"role": "system", "content": protocol})
    return tuple(rendered)


def render_qwen_json_tool_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    enable_thinking: bool = False,
) -> str:
    """Render Qwen with tagged JSON while bypassing its native XML tool mode."""

    return tokenizer.apply_chat_template(
        list(with_qwen_json_tool_protocol(messages, tools)),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


__all__ = [
    "QWEN_JSON_TOOL_PROTOCOL",
    "is_qwen35_family",
    "render_qwen_json_tool_prompt",
    "with_qwen_json_tool_protocol",
]
