import json

import httpx
import pytest

from vllm_agent_gateway.adapters.ollama import (
    messages_to_openai,
    nonstream_response,
    stream_response,
    to_openai,
)


def test_messages_convert_images_calls_and_results():
    messages = messages_to_openai(
        [
            {
                "role": "assistant",
                "content": "look",
                "images": ["data:image/png;base64,abc", "not-valid-base64"],
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "weather", "arguments": {"x": 1}}}
                ],
            },
            {"role": "tool", "tool_name": "weather", "content": {"ok": True}},
        ]
    )

    assert len(messages[0]["content"]) == 3
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"x":1}'
    assert messages[1]["tool_call_id"] == "call-1"


def test_generate_request_maps_options_schema_images_and_thinking():
    request, requested = to_openai(
        {
            "model": "alias",
            "prompt": "inspect",
            "system": "system",
            "images": ["data:image/webp;base64,abc"],
            "stream": True,
            "think": True,
            "format": {"type": "object"},
            "options": {"num_predict": 12, "repeat_penalty": 1.1, "temperature": 0.2},
            "tools": [{"type": "function", "function": {"name": "demo"}}],
        },
        "generate",
        served_model="local",
    )

    assert requested == "alias"
    assert request["max_tokens"] == 12
    assert request["repetition_penalty"] == 1.1
    assert request["response_format"]["type"] == "json_schema"
    assert request["chat_template_kwargs"] == {"enable_thinking": True}
    assert request["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("mode", ["chat", "generate"])
def test_nonstream_response_preserves_reasoning_and_malformed_tool_args(mode):
    result = nonstream_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "reasoning_content": "thought",
                        "tool_calls": [{"function": {"name": "demo", "arguments": "not-json"}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
        "alias",
        mode,
        0,
    )

    assert result["done_reason"] == "tool_calls"
    assert result["prompt_eval_count"] == 3
    if mode == "chat":
        assert result["message"]["thinking"] == "thought"
        assert result["message"]["tool_calls"][0]["function"]["arguments"] == {"raw": "not-json"}
    else:
        assert result["response"] == "answer"
        assert result["thinking"] == "thought"


@pytest.mark.asyncio
async def test_stream_response_emits_content_tools_usage_and_final_chunk():
    events = [
        "not-sse",
        "data: not-json",
        'data: {"choices":[{"delta":{"content":"hi","reasoning":"why",'
        '"tool_calls":[{"index":0,"function":{"name":"demo",'
        '"arguments":"{\\"x\\":1}"}}]}}]}',
        'data: {"usage":{"prompt_tokens":5,"completion_tokens":2},'
        '"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]
    response = httpx.Response(
        200,
        content=("\n\n".join(events) + "\n\n").encode(),
        request=httpx.Request("POST", "http://upstream.test"),
    )

    chunks = [json.loads(chunk) async for chunk in stream_response(response, "local", "chat", 0)]

    assert chunks[0]["message"] == {"role": "assistant", "content": "hi", "thinking": "why"}
    assert chunks[-1]["prompt_eval_count"] == 5
    assert chunks[-1]["message"]["tool_calls"][0]["function"] == {
        "name": "demo",
        "arguments": {"x": 1},
    }
