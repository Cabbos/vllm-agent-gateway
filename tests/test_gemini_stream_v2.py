from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from vllm_agent_gateway.adapters.gemini_stream import (
    convert_openai_sse,
    openai_sse_to_gemini,
    parse_openai_sse,
)


async def _chunks(*chunks: bytes | str) -> AsyncIterator[bytes | str]:
    for chunk in chunks:
        yield chunk


def _sse_event(value: dict) -> bytes:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return f"data: {encoded}\r\n\r\n".encode()


async def _responses(*chunks: bytes | str) -> list[dict]:
    return [response async for response in openai_sse_to_gemini(_chunks(*chunks))]


@pytest.mark.asyncio
async def test_parser_handles_fragmented_unicode_crlf_comments_and_malformed_events():
    valid = _sse_event(
        {
            "id": "chatcmpl-猫",
            "choices": [{"index": 0, "delta": {"content": "你好🌍"}}],
        }
    )
    globe = "🌍".encode()
    split_at = valid.index(globe) + 2
    chunks = (
        b": keepalive\r\n\r\ndata: {bad json}\r\n\r\n",
        valid[:split_at],
        valid[split_at : split_at + 1],
        valid[split_at + 1 : -3],
        valid[-3:-2],
        valid[-2:],
        b"event: ignored\n\ndata: [DONE]\n\n",
        _sse_event({"choices": [{"delta": {"content": "not emitted"}}]}),
    )

    events = [event async for event in parse_openai_sse(_chunks(*chunks))]

    assert len(events) == 1
    assert events[0]["id"] == "chatcmpl-猫"
    assert events[0]["choices"][0]["delta"]["content"] == "你好🌍"


@pytest.mark.asyncio
async def test_text_reasoning_finish_and_usage_are_converted_incrementally():
    raw = b"".join(
        [
            _sse_event(
                {
                    "id": "chatcmpl-1",
                    "model": "Qwen3.5-35B",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "reasoning_content": "先想一想",
                                "content": "答案",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            _sse_event(
                {
                    "id": "chatcmpl-1",
                    "model": "Qwen3.5-35B",
                    "choices": [
                        {"index": 0, "delta": {"content": "是 42。"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 7,
                        "total_tokens": 17,
                        "prompt_tokens_details": {"cached_tokens": 4},
                        "completion_tokens_details": {"reasoning_tokens": 3},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )

    responses = await _responses(raw[:7], raw[7:31], raw[31:103], raw[103:])

    assert responses[0] == {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "先想一想", "thought": True},
                        {"text": "答案"},
                    ],
                    "role": "model",
                },
                "index": 0,
            }
        ],
        "modelVersion": "Qwen3.5-35B",
        "responseId": "chatcmpl-1",
    }
    assert responses[1]["candidates"][0]["content"]["parts"] == [{"text": "是 42。"}]
    assert responses[1]["candidates"][0]["finishReason"] == "STOP"
    assert responses[1]["usageMetadata"] == {
        "promptTokenCount": 10,
        "candidatesTokenCount": 7,
        "totalTokenCount": 17,
        "cachedContentTokenCount": 4,
        "thoughtsTokenCount": 3,
    }


@pytest.mark.asyncio
async def test_fragmented_parallel_tool_calls_are_buffered_until_finish():
    events = [
        {
            "id": "tools-1",
            "model": "local",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-weather",
                                "function": {"name": "get_", "arguments": '{"city":"北'},
                            },
                            {
                                "index": 1,
                                "id": "call-time",
                                "function": {"name": "get_", "arguments": '{"tz":"A'},
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "tools-1",
            "model": "local",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "weather", "arguments": '京"}'}},
                            {"index": 1, "function": {"name": "time", "arguments": 'sia"}'}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    raw = b"".join([*map(_sse_event, events), b"data: [DONE]\n\n"])

    responses = await _responses(*(raw[index : index + 3] for index in range(0, len(raw), 3)))

    assert len(responses) == 1
    candidate = responses[0]["candidates"][0]
    assert candidate["finishReason"] == "STOP"
    calls = [part["functionCall"] for part in candidate["content"]["parts"]]
    assert calls == [
        {
            "name": "get_weather",
            "args": {"city": "北京"},
            "id": "call-weather",
        },
        {
            "name": "get_time",
            "args": {"tz": "Asia"},
            "id": "call-time",
        },
    ]


@pytest.mark.asyncio
async def test_done_flushes_unfinished_tool_and_preserves_invalid_arguments():
    event = {
        "id": "truncated",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"name": "broken", "arguments": '{"open":'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }

    responses = await _responses(_sse_event(event), b"data: [DONE]\n\n")

    call = responses[0]["candidates"][0]["content"]["parts"][0]["functionCall"]
    assert call == {"name": "broken", "args": {"raw": '{"open":'}}


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", ["sse", "json-array"])
async def test_output_framings_round_trip_as_json(framing: str):
    raw = b"".join(
        [
            _sse_event(
                {
                    "id": "frame-1",
                    "model": "模型",
                    "choices": [{"index": 0, "delta": {"content": "甲"}}],
                }
            ),
            _sse_event(
                {
                    "id": "frame-1",
                    "model": "模型",
                    "choices": [
                        {"index": 0, "delta": {"content": "乙"}, "finish_reason": "length"}
                    ],
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )

    encoded = b"".join(
        [chunk async for chunk in convert_openai_sse(_chunks(raw), framing=framing)]
    ).decode()
    if framing == "json-array":
        values = json.loads(encoded)
    else:
        values = [
            json.loads(block.removeprefix("data: ")) for block in encoded.strip().split("\r\n\r\n")
        ]

    assert [item["candidates"][0]["content"]["parts"][0]["text"] for item in values] == [
        "甲",
        "乙",
    ]
    assert values[-1]["candidates"][0]["finishReason"] == "MAX_TOKENS"


@pytest.mark.asyncio
async def test_explicit_identity_overrides_upstream_values():
    event = _sse_event(
        {
            "id": "upstream-id",
            "model": "upstream-model",
            "choices": [{"index": 0, "delta": {"content": "x"}}],
        }
    )

    responses = [
        item
        async for item in openai_sse_to_gemini(
            _chunks(event), model_version="served-model", response_id="gateway-id"
        )
    ]

    assert responses[0]["modelVersion"] == "served-model"
    assert responses[0]["responseId"] == "gateway-id"


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed():
    async def cancelled_source() -> AsyncIterator[bytes]:
        yield _sse_event({"choices": [{"index": 0, "delta": {"content": "first"}}]})
        raise asyncio.CancelledError

    iterator = openai_sse_to_gemini(cancelled_source())
    assert (await anext(iterator))["candidates"][0]["content"]["parts"][0]["text"] == "first"
    with pytest.raises(asyncio.CancelledError):
        await anext(iterator)


@pytest.mark.asyncio
async def test_json_array_is_valid_when_upstream_emits_no_responses():
    encoded = b"".join(
        [
            chunk
            async for chunk in convert_openai_sse(
                _chunks(b": heartbeat\n\ndata: [DONE]\n\n"), framing="json-array"
            )
        ]
    )
    assert encoded == b"[]"


@pytest.mark.asyncio
async def test_unknown_framing_is_rejected():
    with pytest.raises(ValueError, match="framing"):
        await anext(convert_openai_sse(_chunks(b""), framing="ndjson"))  # type: ignore[arg-type]
