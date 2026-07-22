import pytest

from vllm_agent_gateway.adapters.anthropic import transform_request
from vllm_agent_gateway.errors import GatewayError


async def _convert_document(_block):
    return (
        [
            {"type": "text", "text": "document text"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "image"},
            },
        ],
        {"code": "pdf_base64", "pages": 1},
    )


@pytest.mark.asyncio
async def test_transform_normalizes_documents_history_tools_and_metadata():
    payload, events = await transform_request(
        {
            "model": "claude-alias",
            "max_tokens": 16,
            "cache_control": {"type": "ephemeral"},
            "future field": True,
            "tools": [
                {
                    "name": "demo",
                    "description": "demo",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "system": [{"type": "text", "text": "system", "citations": []}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "pdf",
                            },
                        },
                        {"type": "server_tool_use", "name": "search", "input": {"q": "x"}},
                    ],
                }
            ],
        },
        served_model="local",
        convert_document=_convert_document,
        max_prompt_images=1,
    )

    codes = {event["code"] for event in events}
    assert payload["model"] == "local"
    assert "future field" not in payload
    assert "cache_control" not in payload["tools"][0]
    assert [block["type"] for block in payload["messages"][0]["content"]] == [
        "text",
        "image",
        "text",
    ]
    assert {
        "model_routed_local",
        "cache_control_prefix_cache",
        "citations_unavailable",
        "pdf_base64",
        "history_block_to_text",
        "ignored_field_future_field",
    } <= codes


@pytest.mark.asyncio
async def test_nested_tool_result_documents_are_converted_and_compacted():
    payload, events = await transform_request(
        {
            "model": "local",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": "pdf",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        served_model="local",
        convert_document=_convert_document,
        max_prompt_images=0,
    )

    nested = payload["messages"][0]["content"][0]["content"]
    assert [block["type"] for block in nested] == ["text", "text"]
    assert any(event["code"] == "image_history_compacted" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"messages": [], "thinking": "yes"},
            "thinking configuration",
        ),
        (
            {"messages": [], "tools": [{"type": "web_search_2025"}]},
            "cannot run inside local vLLM",
        ),
        (
            {"messages": [{"role": "user", "content": [{"type": "audio", "source": {}}]}]},
            "not supported",
        ),
    ],
)
async def test_invalid_anthropic_controls_fail_explicitly(payload, message):
    with pytest.raises(GatewayError, match=message):
        await transform_request(
            payload,
            served_model="local",
            convert_document=_convert_document,
        )
