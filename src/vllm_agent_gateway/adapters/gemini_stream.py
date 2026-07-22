from __future__ import annotations

import codecs
import json
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

StreamChunk = bytes | str
GeminiFraming = Literal["sse", "json-array"]


@dataclass
class _ToolCall:
    call_id: str | None = None
    name: list[str] = field(default_factory=list)
    arguments: list[str] = field(default_factory=list)


@dataclass
class _SSEEventParser:
    data_lines: list[str] = field(default_factory=list)
    stopped: bool = False

    def consume_line(self, line: str) -> dict[str, Any] | None:
        if line == "":
            if not self.data_lines:
                return None
            payload = "\n".join(self.data_lines)
            self.data_lines.clear()
            if payload.strip() == "[DONE]":
                self.stopped = True
                return None
            try:
                value = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                return None
            return value if isinstance(value, dict) else None
        if line.startswith(":") or ":" not in line:
            return None
        field_name, value = line.split(":", 1)
        if value.startswith(" "):
            value = value[1:]
        if field_name == "data":
            self.data_lines.append(value)
        return None


def _next_line(buffer: str, *, final: bool = False) -> tuple[str, str] | None:
    """Return one SSE line and the unconsumed buffer.

    A trailing CR is retained until the next network chunk so CRLF split across
    chunks is treated as one line ending.
    """

    for index, char in enumerate(buffer):
        if char == "\n":
            return buffer[:index], buffer[index + 1 :]
        if char != "\r":
            continue
        if index + 1 == len(buffer) and not final:
            return None
        consumed = 2 if buffer[index + 1 : index + 2] == "\n" else 1
        return buffer[:index], buffer[index + consumed :]
    if final and buffer:
        return buffer, ""
    return None


async def parse_openai_sse(
    chunks: AsyncIterable[StreamChunk],
) -> AsyncIterator[dict[str, Any]]:
    """Parse OpenAI SSE JSON events from arbitrarily fragmented chunks.

    Comments, unknown fields, malformed JSON events, and empty events are
    ignored. ``data: [DONE]`` terminates the iterator. Unicode decoding and
    cancellation errors intentionally propagate to the caller.
    """

    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    parser = _SSEEventParser()

    async for chunk in chunks:
        if isinstance(chunk, bytes):
            buffer += decoder.decode(chunk, final=False)
        else:
            # Preserve ordering for a source that switches from bytes to text.
            # This also raises on an incomplete UTF-8 byte sequence instead of
            # silently moving the following text ahead of buffered bytes.
            buffer += decoder.decode(b"", final=True) + chunk
            decoder.reset()

        while (line_result := _next_line(buffer)) is not None:
            line, buffer = line_result
            event = parser.consume_line(line)
            if event is not None:
                yield event
            if parser.stopped:
                return

    buffer += decoder.decode(b"", final=True)
    while (line_result := _next_line(buffer, final=True)) is not None:
        line, buffer = line_result
        event = parser.consume_line(line)
        if event is not None:
            yield event
        if parser.stopped:
            return

    # SSE permits the final event to end at EOF without a blank line.
    if parser.data_lines:
        event = parser.consume_line("")
        if event is not None:
            yield event


def _finish_reason(value: Any) -> str:
    return {
        "stop": "STOP",
        "length": "MAX_TOKENS",
        "content_filter": "SAFETY",
        "tool_calls": "STOP",
        "function_call": "STOP",
    }.get(str(value or "").lower(), "OTHER")


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _usage_metadata(usage: Any) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    mapping = {
        "prompt_tokens": "promptTokenCount",
        "completion_tokens": "candidatesTokenCount",
        "total_tokens": "totalTokenCount",
    }
    result: dict[str, int] = {}
    for source, destination in mapping.items():
        if (value := _integer(usage.get(source))) is not None:
            result[destination] = value

    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached = _integer(prompt_details.get("cached_tokens"))
        if cached is not None:
            result["cachedContentTokenCount"] = cached
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning = _integer(completion_details.get("reasoning_tokens"))
        if reasoning is not None:
            result["thoughtsTokenCount"] = reasoning
    return result or None


def _tool_arguments(raw: str) -> Any:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    # Gemini functionCall.args is a Struct, not an arbitrary JSON scalar.
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _tool_parts(calls: dict[int, _ToolCall]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for _, call in sorted(calls.items()):
        function_call: dict[str, Any] = {
            "name": "".join(call.name) or "tool",
            "args": _tool_arguments("".join(call.arguments)),
        }
        if call.call_id:
            function_call["id"] = call.call_id
        parts.append({"functionCall": function_call})
    return parts


def _response_envelope(
    *,
    candidates: list[dict[str, Any]] | None,
    usage: dict[str, int] | None,
    response_id: str,
    model_version: str,
) -> dict[str, Any]:
    response: dict[str, Any] = {}
    if candidates:
        response["candidates"] = candidates
    if usage:
        response["usageMetadata"] = usage
    response["modelVersion"] = model_version
    response["responseId"] = response_id
    return response


def _delta_parts(delta: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    reasoning = next(
        (
            delta[key]
            for key in ("reasoning_content", "reasoning", "thinking")
            if isinstance(delta.get(key), str) and delta[key]
        ),
        None,
    )
    if reasoning is not None:
        parts.append({"text": reasoning, "thought": True})
    content = delta.get("content")
    if isinstance(content, str) and content:
        parts.append({"text": content})
    return parts


def _update_tool_deltas(
    delta: dict[str, Any],
    *,
    choice_index: int,
    tools: dict[int, dict[int, _ToolCall]],
) -> None:
    tool_deltas = delta.get("tool_calls")
    if not isinstance(tool_deltas, list):
        return
    choice_tools = tools.setdefault(choice_index, {})
    for fallback_index, tool_delta in enumerate(tool_deltas):
        if not isinstance(tool_delta, dict):
            continue
        tool_index = _integer(tool_delta.get("index"))
        tool_index = fallback_index if tool_index is None else tool_index
        call = choice_tools.setdefault(tool_index, _ToolCall())
        if tool_delta.get("id") and call.call_id is None:
            call.call_id = str(tool_delta["id"])
        function = tool_delta.get("function")
        if not isinstance(function, dict):
            continue
        if isinstance(function.get("name"), str):
            call.name.append(function["name"])
        if isinstance(function.get("arguments"), str):
            call.arguments.append(function["arguments"])


def _candidate_from_choice(
    choice: Any,
    *,
    fallback_index: int,
    tools: dict[int, dict[int, _ToolCall]],
) -> dict[str, Any] | None:
    if not isinstance(choice, dict):
        return None
    choice_index = _integer(choice.get("index"))
    choice_index = fallback_index if choice_index is None else choice_index
    delta = choice.get("delta")
    delta = delta if isinstance(delta, dict) else {}
    parts = _delta_parts(delta)
    _update_tool_deltas(delta, choice_index=choice_index, tools=tools)

    finish = choice.get("finish_reason")
    if finish is not None:
        parts.extend(_tool_parts(tools.pop(choice_index, {})))
    if not parts and finish is None:
        return None
    candidate: dict[str, Any] = {
        "content": {"parts": parts or [{"text": ""}], "role": "model"},
        "index": choice_index,
    }
    if finish is not None:
        candidate["finishReason"] = _finish_reason(finish)
    return candidate


def _event_candidates(
    event: dict[str, Any], tools: dict[int, dict[int, _ToolCall]]
) -> list[dict[str, Any]]:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return []
    return [
        candidate
        for fallback_index, choice in enumerate(choices)
        if (
            candidate := _candidate_from_choice(
                choice,
                fallback_index=fallback_index,
                tools=tools,
            )
        )
        is not None
    ]


async def openai_sse_to_gemini(
    chunks: AsyncIterable[StreamChunk],
    *,
    model_version: str | None = None,
    response_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Convert OpenAI Chat Completions SSE into Gemini response chunks.

    Tool name and argument deltas are buffered until the corresponding choice
    finishes. The function consumes but never closes ``chunks``; HTTP callers
    should close their upstream response in their own ``finally`` block.
    """

    current_model = model_version or "local-model"
    current_response_id = response_id or "local-gemini-response"
    tools: dict[int, dict[int, _ToolCall]] = {}

    async for event in parse_openai_sse(chunks):
        if model_version is None and event.get("model"):
            current_model = str(event["model"])
        if response_id is None and event.get("id"):
            current_response_id = str(event["id"])

        candidates = _event_candidates(event, tools)

        usage = _usage_metadata(event.get("usage"))
        if candidates or usage:
            yield _response_envelope(
                candidates=candidates,
                usage=usage,
                response_id=current_response_id,
                model_version=current_model,
            )

    # A truncated but otherwise valid stream may reach [DONE]/EOF without a
    # finish_reason. Preserve complete buffered tool calls instead of losing
    # them; invalid/incomplete JSON is exposed under args.raw.
    for choice_index, choice_tools in sorted(tools.items()):
        yield _response_envelope(
            candidates=[
                {
                    "content": {
                        "parts": _tool_parts(choice_tools),
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": choice_index,
                }
            ],
            usage=None,
            response_id=current_response_id,
            model_version=current_model,
        )


async def frame_gemini_responses(
    responses: AsyncIterable[dict[str, Any]],
    *,
    framing: GeminiFraming = "sse",
) -> AsyncIterator[bytes]:
    """Encode Gemini response objects as SSE or a streaming JSON array."""

    if framing not in {"sse", "json-array"}:
        raise ValueError("framing must be 'sse' or 'json-array'")
    if framing == "sse":
        async for response in responses:
            encoded = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
            yield f"data: {encoded}\r\n\r\n".encode()
        return

    yield b"["
    first = True
    async for response in responses:
        if not first:
            yield b","
        first = False
        yield json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode()
    yield b"]"


async def convert_openai_sse(
    chunks: AsyncIterable[StreamChunk],
    *,
    framing: GeminiFraming = "sse",
    model_version: str | None = None,
    response_id: str | None = None,
) -> AsyncIterator[bytes]:
    """Convenience composition used by a streaming HTTP response body."""

    responses = openai_sse_to_gemini(
        chunks,
        model_version=model_version,
        response_id=response_id,
    )
    async for encoded in frame_gemini_responses(responses, framing=framing):
        yield encoded
