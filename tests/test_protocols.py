import json

import pytest

from vllm_agent_gateway.app import (
    SERVED_MODEL,
    AnthropicCompatibilityError,
    _gemini_to_openai,
    _normalize_proxy_path,
    _ollama_nonstream_response,
    _ollama_to_openai,
    _openai_to_gemini,
    transform_anthropic_request,
    transform_openai_request,
)


def test_openai_alias_and_reasoning_are_mapped():
    payload, events = transform_openai_request(
        {
            "model": "gpt-example",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "low",
        }
    )

    assert payload["model"] == SERVED_MODEL
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert events[0]["code"] == "model_routed_local"


def test_openai_responses_reasoning_object_controls_chat_template():
    payload, _events = transform_openai_request(
        {
            "model": "gpt-example",
            "input": "hello",
            "reasoning": {"effort": "none"},
        }
    )

    assert payload["reasoning"] == {"effort": "none"}
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_anthropic_thinking_can_toggle_per_request():
    enabled, enabled_events = transform_anthropic_request(
        {
            "model": "claude-example",
            "max_tokens": 32,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    disabled, disabled_events = transform_anthropic_request(
        {
            "model": "claude-example",
            "max_tokens": 32,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert enabled["chat_template_kwargs"]["enable_thinking"] is True
    assert disabled["chat_template_kwargs"]["enable_thinking"] is False
    assert any(event["code"] == "thinking_enabled" for event in enabled_events)
    assert any(event["code"] == "thinking_disabled" for event in disabled_events)


def test_legacy_transform_wraps_new_adapter_errors():
    with pytest.raises(AnthropicCompatibilityError, match="thinking.type"):
        transform_anthropic_request(
            {
                "model": "claude-example",
                "thinking": {"type": "unexpected"},
                "messages": [{"role": "user", "content": "hello"}],
            }
        )


def test_common_path_aliases():
    assert _normalize_proxy_path("responses") == "v1/responses"
    assert _normalize_proxy_path("openai/v1/chat/completions") == "v1/chat/completions"
    assert (
        _normalize_proxy_path("openai/deployments/demo/chat/completions") == "v1/chat/completions"
    )


def test_ollama_request_and_response_mapping():
    openai, requested = _ollama_to_openai(
        {
            "model": "ollama-alias",
            "stream": False,
            "think": False,
            "messages": [{"role": "user", "content": "hello"}],
            "options": {"num_predict": 16},
        },
        "chat",
    )
    result = _ollama_nonstream_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "weather",
                                    "arguments": json.dumps({"city": "Shanghai"}),
                                }
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        },
        requested,
        "chat",
        0,
    )

    assert openai["model"] == SERVED_MODEL
    assert openai["chat_template_kwargs"]["enable_thinking"] is False
    assert openai["max_tokens"] == 16
    assert result["message"]["tool_calls"][0]["function"]["name"] == "weather"


def test_gemini_function_mapping_round_trip():
    openai = _gemini_to_openai(
        {
            "contents": [{"role": "user", "parts": [{"text": "weather"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ]
                }
            ],
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["weather"],
                }
            },
        }
    )
    gemini = _openai_to_gemini(
        {
            "id": "response-id",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-id",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Shanghai"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    assert openai["tool_choice"]["function"]["name"] == "weather"
    call = gemini["candidates"][0]["content"]["parts"][0]["functionCall"]
    assert call["name"] == "weather"
    assert call["args"] == {"city": "Shanghai"}
