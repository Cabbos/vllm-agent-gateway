from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from ..errors import GatewayError
from .common import CompatibilityEvent, route_to_model

DocumentConverter = Callable[
    [dict[str, Any]], Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]
]


def _file_document(part: dict[str, Any]) -> dict[str, Any]:
    file_info = part.get("file") if isinstance(part.get("file"), dict) else part
    filename = str(file_info.get("filename") or part.get("filename") or "attached PDF")
    if file_info.get("file_id") or part.get("file_id"):
        raise GatewayError(
            "OpenAI file_id inputs require a Files API. Send file_data or file_url instead."
        )
    file_url = file_info.get("file_url") or part.get("file_url")
    if isinstance(file_url, str) and file_url:
        return {
            "type": "document",
            "title": filename,
            "source": {"type": "url", "url": file_url},
        }
    file_data = file_info.get("file_data") or part.get("file_data")
    if not isinstance(file_data, str) or not file_data:
        raise GatewayError("OpenAI input_file/file content requires file_data or file_url.")
    media_type = str(
        file_info.get("media_type")
        or file_info.get("mime_type")
        or part.get("media_type")
        or part.get("mime_type")
        or ""
    ).lower()
    encoded = file_data
    if file_data.startswith("data:"):
        match = re.fullmatch(r"data:([^;,]+)(?:;[^,]*)?;base64,(.*)", file_data, re.DOTALL)
        if not match:
            raise GatewayError("OpenAI file_data data URLs must contain valid base64 data.")
        media_type, encoded = match.group(1).lower(), match.group(2)
    if not media_type:
        lowered = filename.lower()
        media_type = (
            "text/plain"
            if lowered.endswith((".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml"))
            else "application/pdf"
        )
    if media_type not in {"application/pdf", "text/plain"}:
        raise GatewayError(
            f"OpenAI file content type '{media_type}' is not supported locally. "
            "Use PDF/plain text, or send images as input_image/image_url blocks."
        )
    return {
        "type": "document",
        "title": filename,
        "source": {"type": "base64", "media_type": media_type, "data": encoded},
    }


async def _document_replacements(
    part: dict[str, Any], *, responses_style: bool, convert_document: DocumentConverter
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    document = _file_document(part)
    converted, stats = await convert_document(document)
    replacements: list[dict[str, Any]] = []
    for block in converted:
        if block.get("type") == "text":
            replacements.append(
                {
                    "type": "input_text" if responses_style else "text",
                    "text": block.get("text", ""),
                }
            )
            continue
        source = block.get("source") if isinstance(block.get("source"), dict) else {}
        data_url = f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
        replacements.append(
            {"type": "input_image", "image_url": data_url, "detail": "auto"}
            if responses_style
            else {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}}
        )
    stats = dict(stats)
    source_type = document.get("source", {}).get("media_type")
    stats["code"] = (
        "openai_input_file_pdf" if source_type == "application/pdf" else "openai_input_file_text"
    )
    return replacements, stats


async def _convert_parts(
    parts: list[Any], *, responses_style: bool, convert_document: DocumentConverter
) -> tuple[list[Any], list[CompatibilityEvent]]:
    converted: list[Any] = []
    events: list[CompatibilityEvent] = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") in {"input_file", "file"}:
            replacements, stats = await _document_replacements(
                part,
                responses_style=responses_style,
                convert_document=convert_document,
            )
            converted.extend(replacements)
            events.append(stats)
        else:
            converted.append(part)
    return converted, events


async def _convert_documents(
    payload: dict[str, Any],
    events: list[CompatibilityEvent],
    convert_document: DocumentConverter,
) -> None:
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), list):
                message["content"], part_events = await _convert_parts(
                    message["content"],
                    responses_style=False,
                    convert_document=convert_document,
                )
                events.extend(part_events)
    responses_input = payload.get("input")
    if not isinstance(responses_input, list):
        return
    converted_input: list[Any] = []
    for item in responses_input:
        if isinstance(item, dict) and item.get("type") in {"input_file", "file"}:
            replacements, stats = await _document_replacements(
                item,
                responses_style=True,
                convert_document=convert_document,
            )
            converted_input.extend(replacements)
            events.append(stats)
            continue
        if isinstance(item, dict) and isinstance(item.get("content"), list):
            item = dict(item)
            item["content"], part_events = await _convert_parts(
                item["content"],
                responses_style=True,
                convert_document=convert_document,
            )
            events.extend(part_events)
        converted_input.append(item)
    payload["input"] = converted_input


async def transform_request(
    payload: Any,
    *,
    served_model: str,
    convert_document: DocumentConverter,
) -> tuple[Any, list[CompatibilityEvent]]:
    if not isinstance(payload, dict):
        return payload, []
    events: list[CompatibilityEvent] = []
    route_to_model(payload, served_model, events)
    await _convert_documents(payload, events, convert_document)
    effort = payload.get("reasoning_effort")
    reasoning = payload.get("reasoning")
    if effort is None and isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    if isinstance(effort, str):
        kwargs = payload.setdefault("chat_template_kwargs", {})
        if not isinstance(kwargs, dict):
            raise GatewayError("chat_template_kwargs must be an object.")
        kwargs["enable_thinking"] = effort.lower() not in {"none", "disabled"}
    return payload, events
