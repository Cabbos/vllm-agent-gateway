from __future__ import annotations

import base64
import binascii
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from vllm_agent_gateway.errors import GatewayError
from vllm_agent_gateway.proxy.streaming import close_response_shielded


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def model_details(
    *, model_format: str, model_family: str, parameter_size: str, quantization: str
) -> dict[str, Any]:
    return {
        "parent_model": "",
        "format": model_format,
        "family": model_family,
        "families": [model_family],
        "parameter_size": parameter_size,
        "quantization_level": quantization,
    }


def model_entry(
    *,
    served_model: str,
    details: dict[str, Any],
    size_bytes: int,
    vram_bytes: int,
    context_length: int,
    running: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": served_model,
        "model": served_model,
        "modified_at": _utc_now(),
        "size": size_bytes,
        "digest": f"local-{served_model}",
        "details": details,
    }
    if running:
        entry.update(
            {
                "expires_at": (datetime.now(UTC) + timedelta(days=3650))
                .isoformat()
                .replace("+00:00", "Z"),
                "size_vram": vram_bytes,
                "context_length": context_length,
            }
        )
    return entry


def _image_mime_from_base64(encoded: str) -> str:
    if encoded.startswith("data:"):
        return encoded.split(";", 1)[0][5:] or "image/jpeg"
    try:
        prefix = base64.b64decode(encoded[:128] + "===", validate=False)
    except (binascii.Error, ValueError):
        return "image/jpeg"
    if prefix.startswith(b"\x89PNG"):
        return "image/png"
    if prefix.startswith(b"RIFF") and b"WEBP" in prefix[:16]:
        return "image/webp"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return "image/jpeg"


def messages_to_openai(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise GatewayError("Ollama messages must be an array.")
    converted: list[dict[str, Any]] = []
    tool_ids_by_name: dict[str, str] = {}

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        result: dict[str, Any] = {"role": role}
        content = message.get("content", "")
        images = message.get("images")

        if isinstance(images, list) and images:
            parts: list[dict[str, Any]] = []
            if content:
                parts.append({"type": "text", "text": str(content)})
            for encoded in images:
                if not isinstance(encoded, str):
                    continue
                image_url = (
                    encoded
                    if encoded.startswith("data:")
                    else f"data:{_image_mime_from_base64(encoded)};base64,{encoded}"
                )
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
            result["content"] = parts
        else:
            result["content"] = content if isinstance(content, (str, list)) else str(content)

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            openai_calls: list[dict[str, Any]] = []
            for call_index, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    continue
                function_value = call.get("function")
                function = function_value if isinstance(function_value, dict) else call
                name = str(function.get("name") or "tool")
                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                call_id = str(call.get("id") or f"ollama-call-{message_index}-{call_index}")
                tool_ids_by_name[name] = call_id
                openai_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            if openai_calls:
                result["tool_calls"] = openai_calls

        if role == "tool":
            tool_name = str(message.get("tool_name") or "")
            tool_call_id = message.get("tool_call_id") or tool_ids_by_name.get(tool_name)
            if not tool_call_id and tool_ids_by_name:
                tool_call_id = next(reversed(tool_ids_by_name.values()))
            result["tool_call_id"] = str(tool_call_id or f"ollama-tool-result-{message_index}")
        converted.append(result)
    return converted


def to_openai(
    payload: dict[str, Any], mode: str, *, served_model: str
) -> tuple[dict[str, Any], str]:
    requested_model = str(payload.get("model") or served_model)
    options = _as_dict(payload.get("options"))
    if mode == "chat":
        messages = messages_to_openai(payload.get("messages", []))
    else:
        messages = []
        if payload.get("system"):
            messages.append({"role": "system", "content": str(payload["system"])})
        prompt_message: dict[str, Any] = {
            "role": "user",
            "content": str(payload.get("prompt") or ""),
        }
        if isinstance(payload.get("images"), list):
            prompt_message["images"] = payload["images"]
        messages.extend(messages_to_openai([prompt_message]))

    request_payload: dict[str, Any] = {
        "model": served_model,
        "messages": messages,
        "stream": bool(payload.get("stream", True)),
    }
    option_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "seed": "seed",
        "stop": "stop",
        "repeat_penalty": "repetition_penalty",
        "num_predict": "max_tokens",
    }
    for ollama_key, openai_key in option_map.items():
        if ollama_key in options:
            request_payload[openai_key] = options[ollama_key]

    if isinstance(payload.get("tools"), list):
        request_payload["tools"] = payload["tools"]

    output_format = payload.get("format")
    if output_format == "json":
        request_payload["response_format"] = {"type": "json_object"}
    elif isinstance(output_format, dict):
        request_payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ollama_output", "schema": output_format},
        }

    think = payload.get("think")
    if think is not None:
        enabled = bool(think) and str(think).lower() not in {"false", "none", "disabled", "0"}
        request_payload["chat_template_kwargs"] = {"enable_thinking": enabled}
        request_payload["include_reasoning"] = enabled

    if request_payload["stream"]:
        request_payload["stream_options"] = {"include_usage": True}
    return request_payload, requested_model


def _tool_calls_to_ollama(tool_calls: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return converted
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = _as_dict(call.get("function"))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        converted.append(
            {
                "function": {
                    "name": str(function.get("name") or "tool"),
                    "arguments": arguments,
                }
            }
        )
    return converted


def nonstream_response(
    data: dict[str, Any], requested_model: str, mode: str, started_ns: int
) -> dict[str, Any]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice = _as_dict(choices[0] if choices else None)
    message = _as_dict(choice.get("message"))
    usage = _as_dict(data.get("usage"))
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    tool_calls = _tool_calls_to_ollama(message.get("tool_calls"))
    result: dict[str, Any] = {
        "model": requested_model,
        "created_at": _utc_now(),
        "done": True,
        "done_reason": choice.get("finish_reason") or "stop",
        "total_duration": time.perf_counter_ns() - started_ns,
        "load_duration": 0,
        "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
        "eval_count": int(usage.get("completion_tokens") or 0),
    }
    if mode == "chat":
        result["message"] = {"role": "assistant", "content": content}
        if reasoning:
            result["message"]["thinking"] = reasoning
        if tool_calls:
            result["message"]["tool_calls"] = tool_calls
    else:
        result["response"] = content
        if reasoning:
            result["thinking"] = reasoning
    return result


async def stream_response(
    upstream_response: httpx.Response,
    requested_model: str,
    mode: str,
    started_ns: int,
) -> AsyncIterator[str]:
    tool_fragments: dict[int, dict[str, str]] = {}
    finish_reason = "stop"
    usage: dict[str, Any] = {}
    try:
        async for line in upstream_response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event_usage = event.get("usage")
            if isinstance(event_usage, dict):
                usage = event_usage
            choices = event.get("choices") if isinstance(event.get("choices"), list) else []
            if not choices:
                continue
            choice = _as_dict(choices[0])
            delta = _as_dict(choice.get("delta"))
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])

            content = delta.get("content") or ""
            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                index = int(call.get("index") or 0)
                fragment = tool_fragments.setdefault(index, {"name": "", "arguments": ""})
                function = _as_dict(call.get("function"))
                fragment["name"] += str(function.get("name") or "")
                fragment["arguments"] += str(function.get("arguments") or "")

            if content or reasoning:
                chunk: dict[str, Any] = {
                    "model": requested_model,
                    "created_at": _utc_now(),
                    "done": False,
                }
                if mode == "chat":
                    chunk["message"] = {"role": "assistant", "content": content}
                    if reasoning:
                        chunk["message"]["thinking"] = reasoning
                else:
                    chunk["response"] = content
                    if reasoning:
                        chunk["thinking"] = reasoning
                yield json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n"

        final: dict[str, Any] = {
            "model": requested_model,
            "created_at": _utc_now(),
            "done": True,
            "done_reason": finish_reason,
            "total_duration": time.perf_counter_ns() - started_ns,
            "load_duration": 0,
            "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
            "eval_count": int(usage.get("completion_tokens") or 0),
        }
        if mode == "chat":
            final_message: dict[str, Any] = {"role": "assistant", "content": ""}
            if tool_fragments:
                calls: list[dict[str, Any]] = []
                for index in sorted(tool_fragments):
                    fragment = tool_fragments[index]
                    try:
                        arguments = json.loads(fragment["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = {"raw": fragment["arguments"]}
                    calls.append(
                        {"function": {"name": fragment["name"] or "tool", "arguments": arguments}}
                    )
                final_message["tool_calls"] = calls
            final["message"] = final_message
        else:
            final["response"] = ""
        yield json.dumps(final, ensure_ascii=False, separators=(",", ":")) + "\n"
    finally:
        await close_response_shielded(upstream_response)
