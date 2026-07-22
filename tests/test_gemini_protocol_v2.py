from __future__ import annotations

from typing import Any

import pytest

from vllm_agent_gateway.adapters.gemini import from_openai, to_openai
from vllm_agent_gateway.errors import GatewayError


async def _convert_document(
    _block: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raise AssertionError("document conversion was not expected")


async def _validate_image_url(_url: str) -> None:
    raise AssertionError("image validation was not expected")


async def _to_openai(payload: dict[str, Any]) -> dict[str, Any]:
    converted, _events = await to_openai(
        payload,
        served_model="served-model",
        convert_document=_convert_document,
        validate_image_url=_validate_image_url,
    )
    return converted


@pytest.mark.asyncio
async def test_function_response_id_wins_and_same_name_calls_fall_back_in_order():
    payload = {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"id": "call-1", "name": "lookup", "args": {"n": 1}}},
                    {"functionCall": {"id": "call-2", "name": "lookup", "args": {"n": 2}}},
                    {"functionCall": {"id": "call-3", "name": "lookup", "args": {"n": 3}}},
                ],
            },
            {
                "role": "tool",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "call-2",
                            "name": "lookup",
                            "response": {"result": 2},
                        }
                    },
                    {"functionResponse": {"name": "lookup", "response": {"result": 1}}},
                    {"functionResponse": {"name": "lookup", "response": {"result": 3}}},
                ],
            },
        ]
    }

    converted = await _to_openai(payload)

    tool_messages = [message for message in converted["messages"] if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call-2",
        "call-1",
        "call-3",
    ]


@pytest.mark.asyncio
async def test_any_mode_strictly_filters_multiple_allowed_functions():
    converted = await _to_openai(
        {
            "contents": [{"parts": [{"text": "use a tool"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {"name": "alpha"},
                        {"name": "beta"},
                        {"name": "not_allowed"},
                    ]
                }
            ],
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["beta", "alpha"],
                }
            },
        }
    )

    assert [tool["function"]["name"] for tool in converted["tools"]] == ["alpha", "beta"]
    assert converted["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_any_mode_with_one_allowed_function_uses_named_choice_and_filters():
    converted = await _to_openai(
        {
            "contents": [{"parts": [{"text": "use beta"}]}],
            "tools": [{"functionDeclarations": [{"name": "alpha"}, {"name": "beta"}]}],
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["beta"],
                }
            },
        }
    )

    assert [tool["function"]["name"] for tool in converted["tools"]] == ["beta"]
    assert converted["tool_choice"] == {
        "type": "function",
        "function": {"name": "beta"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [None, [], ["unknown"]])
async def test_any_mode_rejects_empty_or_unknown_allowlist(allowed: Any):
    calling: dict[str, Any] = {"mode": "ANY"}
    if allowed is not None:
        calling["allowedFunctionNames"] = allowed
    with pytest.raises(GatewayError, match="allowedFunctionNames"):
        await _to_openai(
            {
                "contents": [{"parts": [{"text": "tool"}]}],
                "tools": [{"functionDeclarations": [{"name": "known"}]}],
                "toolConfig": {"functionCallingConfig": calling},
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("server_tool", [{"googleSearch": {}}, {"codeExecution": {}}])
async def test_server_tools_are_rejected_instead_of_silently_ignored(server_tool: dict[str, Any]):
    with pytest.raises(GatewayError, match="server tool"):
        await _to_openai(
            {
                "contents": [{"parts": [{"text": "search"}]}],
                "tools": [server_tool],
            }
        )


@pytest.mark.asyncio
async def test_count_tokens_generate_content_request_is_unwrapped():
    converted = await _to_openai(
        {
            "generateContentRequest": {
                "contents": [{"role": "user", "parts": [{"text": "wrapped prompt"}]}],
                "generationConfig": {"maxOutputTokens": 12},
            }
        }
    )

    assert converted["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "wrapped prompt"}],
        }
    ]
    assert converted["max_tokens"] == 12


@pytest.mark.asyncio
async def test_count_tokens_rejects_top_level_contents_with_generate_content_request():
    with pytest.raises(GatewayError, match="mutually exclusive"):
        await _to_openai(
            {
                "contents": [],
                "generateContentRequest": {"contents": []},
            }
        )


@pytest.mark.asyncio
async def test_count_tokens_requires_generate_content_request_object():
    with pytest.raises(GatewayError, match="must be an object"):
        await _to_openai({"generateContentRequest": []})


def test_nonstream_wraps_non_object_tool_args_and_maps_detailed_usage():
    converted = from_openai(
        {
            "id": "response-1",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-scalar",
                                "function": {"name": "scalar", "arguments": "7"},
                            },
                            {
                                "id": "call-list",
                                "function": {"name": "items", "arguments": [1, 2]},
                            },
                            {
                                "id": "call-bad",
                                "function": {"name": "broken", "arguments": "{bad"},
                            },
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        },
        served_model="served-model",
    )

    calls = [part["functionCall"] for part in converted["candidates"][0]["content"]["parts"]]
    assert [call["args"] for call in calls] == [
        {"value": 7},
        {"value": [1, 2]},
        {"raw": "{bad"},
    ]
    assert converted["usageMetadata"] == {
        "promptTokenCount": 10,
        "candidatesTokenCount": 8,
        "totalTokenCount": 18,
        "cachedContentTokenCount": 3,
        "thoughtsTokenCount": 4,
    }
