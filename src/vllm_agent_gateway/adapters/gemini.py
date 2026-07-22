from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from vllm_agent_gateway.errors import GatewayError

from .common import CompatibilityEvent
from .gemini_stream import _tool_arguments, _usage_metadata

DocumentConverter = Callable[
    [dict[str, Any]],
    Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]],
]
ImageURLValidator = Callable[[str], Awaitable[None]]

_FUNCTION_DECLARATION_KEYS = {"functionDeclarations", "function_declarations"}


def model_metadata(
    *,
    served_model: str,
    context_length: int,
    quantization: str,
) -> dict[str, Any]:
    return {
        "name": f"models/{served_model}",
        "baseModelId": served_model,
        "version": f"local-{quantization.lower()}",
        "displayName": f"Local {served_model}",
        "description": "Local model served through vLLM Agent Gateway.",
        "inputTokenLimit": context_length,
        "outputTokenLimit": min(context_length, 32768),
        "supportedGenerationMethods": [
            "generateContent",
            "streamGenerateContent",
            "countTokens",
        ],
    }


def _document_to_openai(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") == "text":
            output.append({"type": "text", "text": block.get("text", "")})
            continue
        source_value = block.get("source")
        source = source_value if isinstance(source_value, dict) else {}
        output.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{source.get('media_type', 'image/jpeg')};base64,"
                        f"{source.get('data', '')}"
                    )
                },
            }
        )
    return output


async def _inline_part(
    part: dict[str, Any], convert_document: DocumentConverter
) -> tuple[list[dict[str, Any]], CompatibilityEvent | None]:
    inline = part.get("inlineData") or part.get("inline_data")
    if not isinstance(inline, dict):
        return [], None
    media_type = str(inline.get("mimeType") or inline.get("mime_type") or "").lower()
    data = inline.get("data")
    if not isinstance(data, str):
        raise GatewayError("Gemini inlineData.data must be a base64 string.")
    if media_type.startswith("image/"):
        return [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
        ], None
    if media_type in {"application/pdf", "text/plain"}:
        document = {
            "type": "document",
            "title": str(part.get("displayName") or "Gemini inline document"),
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
        converted, stats = await convert_document(document)
        return _document_to_openai(converted), stats
    raise GatewayError(
        f"Gemini inlineData type '{media_type or '<missing>'}' is not supported by this model. "
        "Use images, PDF, or plain text; transcribe audio/video before sending."
    )


async def _file_part(
    part: dict[str, Any],
    convert_document: DocumentConverter,
    validate_image_url: ImageURLValidator,
) -> tuple[list[dict[str, Any]], CompatibilityEvent | None]:
    file_data = part.get("fileData") or part.get("file_data")
    if not isinstance(file_data, dict):
        return [], None
    media_type = str(file_data.get("mimeType") or file_data.get("mime_type") or "").lower()
    uri = file_data.get("fileUri") or file_data.get("file_uri")
    if not isinstance(uri, str):
        raise GatewayError("Gemini fileData.fileUri must be a URL.")
    if uri.startswith("gs://") or uri.startswith("files/"):
        raise GatewayError(
            "Gemini Files/Google Cloud URIs require Google's Files API. "
            "Send inlineData or an explicitly allowed HTTPS URL."
        )
    if media_type.startswith("image/"):
        await validate_image_url(uri)
        return [{"type": "image_url", "image_url": {"url": uri}}], None
    if media_type in {"application/pdf", "text/plain"}:
        document = {
            "type": "document",
            "title": str(part.get("displayName") or "Gemini URL document"),
            "source": {"type": "url", "media_type": media_type, "url": uri},
        }
        converted, stats = await convert_document(document)
        return _document_to_openai(converted), stats
    raise GatewayError(
        f"Gemini fileData type '{media_type or '<missing>'}' is not supported locally."
    )


def _unwrap_request(payload: dict[str, Any]) -> dict[str, Any]:
    nested_keys = [
        key for key in ("generateContentRequest", "generate_content_request") if key in payload
    ]
    if len(nested_keys) > 1:
        raise GatewayError(
            "Gemini countTokens must not include both generateContentRequest aliases."
        )
    if not nested_keys:
        return payload
    if "contents" in payload:
        raise GatewayError(
            "Gemini countTokens contents and generateContentRequest are mutually exclusive."
        )
    nested_payload = payload[nested_keys[0]]
    if not isinstance(nested_payload, dict):
        raise GatewayError("Gemini generateContentRequest must be an object.")
    return nested_payload


def _system_message(payload: dict[str, Any]) -> dict[str, Any] | None:
    instruction = payload.get("systemInstruction") or payload.get("system_instruction")
    if not isinstance(instruction, dict):
        return None
    text = "\n".join(
        str(part.get("text"))
        for part in instruction.get("parts", [])
        if isinstance(part, dict) and part.get("text") is not None
    )
    return {"role": "system", "content": text} if text else None


def _function_call(
    part: dict[str, Any],
    *,
    content_index: int,
    part_index: int,
    tool_ids: defaultdict[str, deque[str]],
) -> dict[str, Any] | None:
    function_call = part.get("functionCall") or part.get("function_call")
    if not isinstance(function_call, dict):
        return None
    name = str(function_call.get("name") or "tool")
    call_id = str(function_call.get("id") or f"gemini-call-{content_index}-{part_index}")
    tool_ids[name].append(call_id)
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                function_call.get("args") or {},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def _function_response_message(
    response: dict[str, Any],
    *,
    content_index: int,
    response_index: int,
    tool_ids: defaultdict[str, deque[str]],
) -> dict[str, Any]:
    name = str(response.get("name") or "tool")
    explicit_id = response.get("id")
    response_id = str(explicit_id) if explicit_id else None
    pending_ids = tool_ids[name]
    if response_id is not None:
        with suppress(ValueError):
            pending_ids.remove(response_id)
    elif pending_ids:
        response_id = pending_ids.popleft()
    else:
        response_id = f"gemini-result-{content_index}-{response_index}"
    value = response.get("response", {})
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )
    return {"role": "tool", "tool_call_id": response_id, "content": text}


async def _content_messages(
    content: dict[str, Any],
    *,
    content_index: int,
    tool_ids: defaultdict[str, deque[str]],
    convert_document: DocumentConverter,
    validate_image_url: ImageURLValidator,
) -> tuple[list[dict[str, Any]], list[CompatibilityEvent]]:
    role = "assistant" if content.get("role") == "model" else "user"
    openai_parts: list[dict[str, Any]] = []
    function_calls: list[dict[str, Any]] = []
    function_responses: list[dict[str, Any]] = []
    events: list[CompatibilityEvent] = []
    for part_index, part in enumerate(content.get("parts") or []):
        if not isinstance(part, dict):
            continue
        if part.get("text") is not None:
            openai_parts.append({"type": "text", "text": str(part["text"])})
            continue
        inline_blocks, inline_event = await _inline_part(part, convert_document)
        file_blocks, file_event = await _file_part(part, convert_document, validate_image_url)
        openai_parts.extend(inline_blocks)
        openai_parts.extend(file_blocks)
        events.extend(event for event in (inline_event, file_event) if event is not None)
        if call := _function_call(
            part,
            content_index=content_index,
            part_index=part_index,
            tool_ids=tool_ids,
        ):
            function_calls.append(call)
        response = part.get("functionResponse") or part.get("function_response")
        if isinstance(response, dict):
            function_responses.append(response)

    messages: list[dict[str, Any]] = []
    if openai_parts or function_calls:
        message: dict[str, Any] = {"role": role, "content": openai_parts or ""}
        if function_calls:
            message.update(role="assistant", tool_calls=function_calls)
        messages.append(message)
    messages.extend(
        _function_response_message(
            response,
            content_index=content_index,
            response_index=response_index,
            tool_ids=tool_ids,
        )
        for response_index, response in enumerate(function_responses)
    )
    return messages, events


async def _messages(
    payload: dict[str, Any],
    *,
    convert_document: DocumentConverter,
    validate_image_url: ImageURLValidator,
) -> tuple[list[dict[str, Any]], list[CompatibilityEvent]]:
    messages = [message] if (message := _system_message(payload)) else []
    events: list[CompatibilityEvent] = []
    tool_ids: defaultdict[str, deque[str]] = defaultdict(deque)
    contents = payload.get("contents")
    if not isinstance(contents, list):
        raise GatewayError("Gemini contents must be an array.")
    for content_index, content in enumerate(contents):
        if not isinstance(content, dict):
            continue
        converted, content_events = await _content_messages(
            content,
            content_index=content_index,
            tool_ids=tool_ids,
            convert_document=convert_document,
            validate_image_url=validate_image_url,
        )
        messages.extend(converted)
        events.extend(content_events)
    return messages, events


def _apply_generation_config(payload: dict[str, Any], request: dict[str, Any]) -> None:
    generation = payload.get("generationConfig") or payload.get("generation_config")
    if not isinstance(generation, dict):
        return
    generation_map = {
        "temperature": "temperature",
        "topP": "top_p",
        "topK": "top_k",
        "maxOutputTokens": "max_tokens",
        "stopSequences": "stop",
        "seed": "seed",
    }
    for gemini_key, openai_key in generation_map.items():
        if gemini_key in generation:
            request[openai_key] = generation[gemini_key]
    mime = generation.get("responseMimeType")
    schema = generation.get("responseSchema") or generation.get("responseJsonSchema")
    if schema:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "gemini_output", "schema": schema},
        }
    elif mime == "application/json":
        request["response_format"] = {"type": "json_object"}
    thinking = generation.get("thinkingConfig") or generation.get("thinking_config")
    if isinstance(thinking, dict):
        budget = thinking.get("thinkingBudget", thinking.get("thinking_budget"))
        level = str(thinking.get("thinkingLevel") or thinking.get("thinking_level") or "")
        enabled = budget != 0 and level.lower() not in {"minimal", "none", "disabled"}
        request["chat_template_kwargs"] = {"enable_thinking": enabled}
        request["include_reasoning"] = enabled and bool(
            thinking.get("includeThoughts", thinking.get("include_thoughts", True))
        )


def _function_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for group in payload.get("tools") or []:
        if not isinstance(group, dict):
            raise GatewayError("Each Gemini tools entry must be an object.")
        unsupported = sorted(set(group) - _FUNCTION_DECLARATION_KEYS)
        if unsupported:
            names = ", ".join(unsupported)
            raise GatewayError(
                f"Gemini server tool fields ({names}) cannot run inside local vLLM. "
                "Expose them as client-executed functionDeclarations instead."
            )
        declaration_keys = _FUNCTION_DECLARATION_KEYS.intersection(group)
        if not declaration_keys:
            raise GatewayError(
                "Gemini tools entries must contain functionDeclarations; server tools are "
                "not available locally."
            )
        declarations = (
            group.get("functionDeclarations")
            if "functionDeclarations" in group
            else group.get("function_declarations")
        )
        if not isinstance(declarations, list):
            raise GatewayError("Gemini functionDeclarations must be an array.")
        for declaration in declarations:
            if not isinstance(declaration, dict):
                raise GatewayError("Each Gemini function declaration must be an object.")
            function = {
                "name": str(declaration.get("name") or "tool"),
                "description": str(declaration.get("description") or ""),
                "parameters": declaration.get("parameters")
                or declaration.get("parametersJsonSchema")
                or {"type": "object", "properties": {}},
            }
            tools.append({"type": "function", "function": function})
    return tools


def _apply_tool_config(
    payload: dict[str, Any], request: dict[str, Any], tools: list[dict[str, Any]]
) -> None:
    tool_config = payload.get("toolConfig") or payload.get("tool_config") or {}
    calling = (
        tool_config.get("functionCallingConfig") or tool_config.get("function_calling_config") or {}
        if isinstance(tool_config, dict)
        else {}
    )
    mode = str(calling.get("mode") or "AUTO").upper() if isinstance(calling, dict) else "AUTO"
    if mode != "ANY":
        if tools and mode != "NONE":
            request.update(tools=tools, tool_choice="auto")
        return

    allowed = (
        calling.get("allowedFunctionNames") or calling.get("allowed_function_names")
        if isinstance(calling, dict)
        else None
    )
    if not isinstance(allowed, list) or not allowed:
        raise GatewayError("Gemini ANY mode requires a non-empty allowedFunctionNames array.")
    allowed_names = [str(name) for name in allowed if str(name)]
    if not allowed_names:
        raise GatewayError("Gemini ANY mode requires a non-empty allowedFunctionNames array.")
    declared_names = {str(tool["function"]["name"]) for tool in tools}
    unknown_names = sorted(set(allowed_names) - declared_names)
    if unknown_names:
        raise GatewayError(
            "Gemini allowedFunctionNames contains undeclared functions: "
            + ", ".join(unknown_names)
            + "."
        )
    allowed_set = set(allowed_names)
    selected_tools = [tool for tool in tools if str(tool["function"]["name"]) in allowed_set]
    unique_allowed = list(dict.fromkeys(allowed_names))
    request["tools"] = selected_tools
    request["tool_choice"] = (
        {"type": "function", "function": {"name": unique_allowed[0]}}
        if len(unique_allowed) == 1
        else "required"
    )


async def to_openai(
    payload: dict[str, Any],
    *,
    served_model: str,
    convert_document: DocumentConverter,
    validate_image_url: ImageURLValidator,
) -> tuple[dict[str, Any], list[CompatibilityEvent]]:
    payload = _unwrap_request(payload)
    messages, events = await _messages(
        payload,
        convert_document=convert_document,
        validate_image_url=validate_image_url,
    )
    request: dict[str, Any] = {
        "model": served_model,
        "messages": messages,
        "stream": False,
    }
    _apply_generation_config(payload, request)
    _apply_tool_config(payload, request, _function_tools(payload))
    return request, events


def from_openai(data: dict[str, Any], *, served_model: str) -> dict[str, Any]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice_value = choices[0] if choices else None
    choice = choice_value if isinstance(choice_value, dict) else {}
    message_value = choice.get("message")
    message = message_value if isinstance(message_value, dict) else {}
    parts: list[dict[str, Any]] = []
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        parts.append({"text": str(reasoning), "thought": True})
    if message.get("content"):
        parts.append({"text": str(message["content"])})
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function_value = tool_call.get("function")
        function = function_value if isinstance(function_value, dict) else {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = _tool_arguments(arguments)
        elif arguments is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            arguments = {"value": arguments}
        parts.append(
            {
                "functionCall": {
                    "id": tool_call.get("id"),
                    "name": str(function.get("name") or "tool"),
                    "args": arguments,
                }
            }
        )
    finish_map = {
        "stop": "STOP",
        "length": "MAX_TOKENS",
        "content_filter": "SAFETY",
        "tool_calls": "STOP",
    }
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    usage_metadata = _usage_metadata(usage) or {
        "promptTokenCount": 0,
        "candidatesTokenCount": 0,
        "totalTokenCount": 0,
    }
    return {
        "candidates": [
            {
                "content": {"parts": parts or [{"text": ""}], "role": "model"},
                "finishReason": finish_map.get(str(choice.get("finish_reason") or "stop"), "STOP"),
                "index": 0,
            }
        ],
        "usageMetadata": usage_metadata,
        "modelVersion": served_model,
        "responseId": str(data.get("id") or "local-gemini-response"),
    }
