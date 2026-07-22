from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from ..errors import GatewayError
from .common import CompatibilityEvent, route_to_model

DocumentConverter = Callable[
    [dict[str, Any]], Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]
]

VLLM_ANTHROPIC_FIELDS = {
    "model",
    "messages",
    "max_tokens",
    "metadata",
    "output_config",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "kv_transfer_params",
    "chat_template_kwargs",
}

VLLM_CONTENT_TYPES = {
    "text",
    "image",
    "tool_use",
    "tool_result",
    "tool_reference",
    "thinking",
    "redacted_thinking",
}

HISTORY_ONLY_CONTENT_TYPES = {
    "search_result",
    "server_tool_use",
    "web_search_tool_result",
    "web_fetch_tool_result",
    "code_execution_tool_result",
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "tool_search_tool_result",
    "container_upload",
}

SERVER_TOOL_PREFIXES = (
    "web_search_",
    "web_fetch_",
    "code_execution_",
    "computer_",
    "text_editor_",
    "bash_",
    "tool_search_",
    "memory_",
    "mcp_toolset",
)


def _readable_json(value: Any, max_chars: int = 100_000) -> str:
    def sanitize(item: Any) -> Any:
        if isinstance(item, dict):
            clean: dict[str, Any] = {}
            for key, child in item.items():
                if key in {"cache_control", "citations"}:
                    continue
                if (
                    key in {"encrypted_content", "data"}
                    and isinstance(child, str)
                    and len(child) > 512
                ):
                    clean[key] = f"[opaque payload omitted: {len(child)} characters]"
                else:
                    clean[key] = sanitize(child)
            return clean
        if isinstance(item, list):
            return [sanitize(child) for child in item]
        return item

    rendered = json.dumps(sanitize(value), ensure_ascii=False, indent=2, default=str)
    return (
        rendered[:max_chars] + "\n[Historical tool content truncated.]"
        if len(rendered) > max_chars
        else rendered
    )


def _history_block_to_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type") or "unknown")
    if block_type == "search_result":
        title = str(block.get("title") or "Search result")
        source = str(block.get("source") or "")
        content = block.get("content")
        fragments: list[str] = []
        if isinstance(content, str):
            fragments.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    fragments.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    fragments.append(item["text"])
                else:
                    fragments.append(_readable_json(item))
        elif content is not None:
            fragments.append(_readable_json(content))
        header = f"[Search result: {title}]" + (f"\nSource: {source}" if source else "")
        return header + ("\n" + "\n".join(fragments) if fragments else "")
    if block_type == "server_tool_use":
        name = block.get("name") or "server_tool"
        return f"[Historical Anthropic server tool call: {name}]\n{_readable_json(block.get('input', {}))}"
    return f"[Historical Anthropic block converted to text: {block_type}]\n{_readable_json(block)}"


def _apply_thinking(payload: dict[str, Any], events: list[CompatibilityEvent]) -> None:
    if "thinking" not in payload:
        return
    thinking = payload.pop("thinking")
    if not isinstance(thinking, dict):
        raise GatewayError("Anthropic thinking configuration must be an object.")
    thinking_type = thinking.get("type")
    if thinking_type in {"enabled", "adaptive"}:
        enabled = True
    elif thinking_type == "disabled":
        enabled = False
    else:
        raise GatewayError("Anthropic thinking.type must be 'enabled', 'adaptive', or 'disabled'.")
    kwargs = payload.setdefault("chat_template_kwargs", {})
    if not isinstance(kwargs, dict):
        raise GatewayError("chat_template_kwargs must be an object.")
    kwargs["enable_thinking"] = enabled
    events.append({"code": "thinking_enabled" if enabled else "thinking_disabled"})
    if thinking.get("display") == "omitted":
        events.append({"code": "thinking_display_not_filtered"})


def _normalize_tools(payload: dict[str, Any], events: list[CompatibilityEvent]) -> None:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    normalized: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            normalized.append(tool)
            continue
        tool = dict(tool)
        if "cache_control" in tool:
            tool.pop("cache_control", None)
            events.append({"code": "cache_control_prefix_cache"})
        tool_type = str(tool.get("type") or "")
        if tool_type and (tool_type == "mcp_toolset" or tool_type.startswith(SERVER_TOOL_PREFIXES)):
            raise GatewayError(
                f"Anthropic server tool '{tool_type}' cannot run inside local vLLM. "
                "Expose it as a client tool or MCP tool in the calling agent."
            )
        normalized.append(tool)
    payload["tools"] = normalized


async def _convert_blocks(
    blocks: list[Any], convert_document: DocumentConverter
) -> tuple[list[Any], list[CompatibilityEvent]]:
    converted: list[Any] = []
    events: list[CompatibilityEvent] = []
    for original in blocks:
        if not isinstance(original, dict):
            converted.append(original)
            continue
        block = dict(original)
        if "cache_control" in block:
            block.pop("cache_control", None)
            events.append({"code": "cache_control_prefix_cache"})
        if "citations" in block:
            block.pop("citations", None)
            events.append({"code": "citations_unavailable"})
        if block.get("type") == "document":
            replacements, stats = await convert_document(block)
            converted.extend(replacements)
            events.append(stats)
            continue
        block_type = block.get("type")
        if block_type in HISTORY_ONLY_CONTENT_TYPES:
            converted.append({"type": "text", "text": _history_block_to_text(block)})
            events.append({"code": "history_block_to_text", "block_type": block_type})
            continue
        if block_type == "tool_result" and isinstance(block.get("content"), list):
            block["content"], nested_events = await _convert_blocks(
                block["content"], convert_document
            )
            events.extend(nested_events)
        if block_type not in VLLM_CONTENT_TYPES:
            raise GatewayError(
                f"Anthropic content block type '{block_type}' is not supported by the local endpoint."
            )
        converted.append(block)
    return converted, events


def _image_locations(blocks: Any) -> Iterator[tuple[list[Any], int]]:
    if not isinstance(blocks, list):
        return
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "image":
            yield blocks, index
        elif block.get("type") == "tool_result":
            yield from _image_locations(block.get("content"))


def _compact_image_history(
    payload: dict[str, Any],
    *,
    max_prompt_images: int,
    events: list[CompatibilityEvent],
) -> None:
    """Replace the oldest images after the prompt-wide budget is exhausted."""
    locations = [
        location
        for message in payload.get("messages", [])
        if isinstance(message, dict)
        for location in _image_locations(message.get("content"))
    ]
    omitted_count = max(0, len(locations) - max_prompt_images)
    if omitted_count == 0:
        return

    placeholder = (
        "[Earlier image omitted by the local gateway to stay within the "
        f"{max_prompt_images}-image prompt limit. Its visual data is not "
        "available in this turn.]"
    )
    for blocks, index in locations[:omitted_count]:
        blocks[index] = {"type": "text", "text": placeholder}

    events.append(
        {
            "code": "image_history_compacted",
            "images_seen": len(locations),
            "images_retained": len(locations) - omitted_count,
            "images_omitted": omitted_count,
        }
    )


async def transform_request(
    payload: Any,
    *,
    served_model: str,
    convert_document: DocumentConverter,
    max_prompt_images: int = 4,
) -> tuple[Any, list[CompatibilityEvent]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return payload, []
    events: list[CompatibilityEvent] = []
    route_to_model(payload, served_model, events)
    _apply_thinking(payload, events)
    if "cache_control" in payload:
        payload.pop("cache_control", None)
        events.append({"code": "cache_control_prefix_cache"})
    for key in list(payload):
        if key not in VLLM_ANTHROPIC_FIELDS:
            payload.pop(key, None)
            safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
            events.append({"code": f"ignored_field_{safe_key}"})
    _normalize_tools(payload, events)
    if isinstance(payload.get("system"), list):
        payload["system"], block_events = await _convert_blocks(payload["system"], convert_document)
        events.extend(block_events)
    for message in payload["messages"]:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            message["content"], block_events = await _convert_blocks(
                message["content"], convert_document
            )
            events.extend(block_events)
    _compact_image_history(
        payload,
        max_prompt_images=max_prompt_images,
        events=events,
    )
    return payload, events
