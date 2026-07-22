#!/usr/bin/env python3
"""Unified local-agent compatibility gateway in front of vLLM.

The gateway exposes OpenAI/Responses, Anthropic Messages, Ollama, Azure-style,
and Gemini-style request surfaces; routes model aliases to one local model;
maps client-native thinking controls; and converts PDF documents to text/images.
Unknown endpoints are forwarded to vLLM unchanged.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import re
import secrets
import socket
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urljoin, urlsplit

import anyio
import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import settings

UPSTREAM = settings.upstream
SERVED_MODEL = settings.served_model
MODEL_CONTEXT_LENGTH = settings.model_context_length
MAX_PROMPT_IMAGES = settings.max_prompt_images
MAX_PDF_BYTES = settings.max_pdf_bytes
MAX_PDF_PAGES = settings.max_pdf_pages
MAX_RENDERED_PAGES = settings.max_rendered_pages
MAX_EXTRACTED_CHARS = settings.max_extracted_chars
TEXT_PAGE_THRESHOLD = 40
MAX_HISTORY_BLOCK_CHARS = 100_000
MAX_URL_REDIRECTS = 5
TRANSPARENT_PROXY_FAKE_IP_RANGES = (ipaddress.ip_network("198.18.0.0/15"),)

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

# These blocks may appear when a conversation started on Claude with Anthropic
# server tools and is later continued through this local endpoint. Qwen cannot
# replay those server tools, but preserving their visible history as text is
# much better than rejecting the entire conversation.
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

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


class AnthropicCompatibilityError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _safe_title(block: dict[str, Any]) -> str:
    title = block.get("title") or block.get("name") or "attached PDF"
    title = re.sub(r"[\r\n\t]+", " ", str(title)).strip()
    return title[:200] or "attached PDF"


def _check_size(raw: bytes, label: str = "PDF") -> bytes:
    if len(raw) > MAX_PDF_BYTES:
        raise AnthropicCompatibilityError(
            f"{label} is {len(raw) / 1024 / 1024:.1f} MiB; the local limit is "
            f"{MAX_PDF_BYTES / 1024 / 1024:.0f} MiB.",
            status_code=413,
        )
    return raw


def _validate_public_document_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise AnthropicCompatibilityError("Document URL must use http:// or https://.")
    if parsed.username or parsed.password:
        raise AnthropicCompatibilityError("Credentials are not allowed in document URLs.")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(
        (".localhost", ".local", ".internal", ".lan", ".home", ".arpa")
    ):
        raise AnthropicCompatibilityError(
            "Local or internal hostnames are not allowed in document URLs."
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise AnthropicCompatibilityError("Document URL contains an invalid port.") from exc
    if port not in {None, 80, 443}:
        raise AnthropicCompatibilityError("Document URLs may only use ports 80 or 443.")

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname, port or (443 if parsed.scheme == "https" else 80)
            )
        }
    except socket.gaierror as exc:
        raise AnthropicCompatibilityError(f"Unable to resolve document URL host: {exc}.") from exc
    if not addresses:
        raise AnthropicCompatibilityError("Document URL host did not resolve to an address.")

    def address_is_allowed(address: str) -> bool:
        ip = ipaddress.ip_address(address)
        return ip.is_global or any(ip in network for network in TRANSPARENT_PROXY_FAKE_IP_RANGES)

    if any(not address_is_allowed(address) for address in addresses):
        raise AnthropicCompatibilityError(
            "Document URL resolves to a private, loopback, link-local, or reserved address. "
            "Only public document URLs are allowed."
        )


def _fetch_public_document(url: str) -> bytes:
    current_url = url
    timeout = httpx.Timeout(30.0, connect=10.0)
    with httpx.Client(
        timeout=timeout,
        follow_redirects=False,
        headers={"user-agent": "local-anthropic-compat/1.0"},
    ) as client:
        for _ in range(MAX_URL_REDIRECTS + 1):
            _validate_public_document_url(current_url)
            try:
                with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise AnthropicCompatibilityError(
                                "Document URL redirect did not include a Location header."
                            )
                        current_url = urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    declared_size = response.headers.get("content-length")
                    if declared_size:
                        try:
                            too_large = int(declared_size) > MAX_PDF_BYTES
                        except ValueError:
                            too_large = False
                        if too_large:
                            raise AnthropicCompatibilityError(
                                f"Remote document exceeds the local {MAX_PDF_BYTES / 1024 / 1024:.0f} MiB limit.",
                                status_code=413,
                            )
                    chunks = bytearray()
                    for chunk in response.iter_bytes():
                        chunks.extend(chunk)
                        if len(chunks) > MAX_PDF_BYTES:
                            raise AnthropicCompatibilityError(
                                f"Remote document exceeds the local {MAX_PDF_BYTES / 1024 / 1024:.0f} MiB limit.",
                                status_code=413,
                            )
                    return bytes(chunks)
            except httpx.HTTPStatusError as exc:
                raise AnthropicCompatibilityError(
                    f"Document URL returned HTTP {exc.response.status_code}."
                ) from exc
            except httpx.RequestError as exc:
                raise AnthropicCompatibilityError(f"Unable to fetch document URL: {exc}.") from exc
        raise AnthropicCompatibilityError(
            f"Document URL exceeded the {MAX_URL_REDIRECTS}-redirect limit."
        )


def _decode_base64_data(encoded: Any, label: str) -> bytes:
    if not isinstance(encoded, str):
        raise AnthropicCompatibilityError(f"{label} source.data must be a base64 string.")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AnthropicCompatibilityError(f"{label} source.data is not valid base64.") from exc


def _load_pdf_source(block: dict[str, Any]) -> tuple[bytes, str]:
    source = block.get("source")
    if not isinstance(source, dict):
        raise AnthropicCompatibilityError("PDF document block is missing a valid source object.")

    source_type = source.get("type")
    media_type = str(source.get("media_type", "")).lower()
    if source_type == "base64" and media_type == "application/pdf":
        return _check_size(_decode_base64_data(source.get("data"), "PDF")), "pdf_base64"
    if source_type == "url" and isinstance(source.get("url"), str):
        return _check_size(_fetch_public_document(source["url"])), "pdf_url"
    if source_type == "file":
        raise AnthropicCompatibilityError(
            "Anthropic file_id document sources require the Files API, which is not available "
            "on this local endpoint. Send the PDF as base64 or a public HTTPS URL."
        )
    raise AnthropicCompatibilityError(
        "PDF documents must use a base64 application/pdf source or a public HTTP(S) URL."
    )


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(data: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def convert_pdf_document(block: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw, source_kind = _load_pdf_source(block)
    title = _safe_title(block)

    try:
        document = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise AnthropicCompatibilityError(f"Unable to open PDF: {exc}") from exc

    try:
        if document.needs_pass:
            raise AnthropicCompatibilityError("Password-protected PDFs are not supported.")
        if document.page_count == 0:
            raise AnthropicCompatibilityError("PDF contains no pages.")
        if document.page_count > MAX_PDF_PAGES:
            raise AnthropicCompatibilityError(
                f"PDF has {document.page_count} pages; the local limit is {MAX_PDF_PAGES}. "
                "Split the document into smaller parts.",
                status_code=413,
            )

        output: list[dict[str, Any]] = []
        extracted_chars = 0
        rendered_pages = 0
        total_pages = document.page_count

        for page_index in range(total_pages):
            page = document.load_page(page_index)
            text = page.get_text("text", sort=True).replace("\x00", "").strip()
            meaningful_chars = len(re.sub(r"\s+", "", text))
            page_label = f"[PDF: {title} — page {page_index + 1}/{total_pages}]"

            if meaningful_chars >= TEXT_PAGE_THRESHOLD:
                remaining = MAX_EXTRACTED_CHARS - extracted_chars
                if remaining <= 0:
                    output.append(
                        _text_block(
                            f"{page_label}\n[Text omitted because the PDF extraction limit "
                            "was reached.]"
                        )
                    )
                    continue
                page_text = text[:remaining]
                extracted_chars += len(page_text)
                suffix = "\n[Page text truncated.]" if len(page_text) < len(text) else ""
                output.append(_text_block(f"{page_label}\n{page_text}{suffix}"))
                continue

            if rendered_pages >= MAX_RENDERED_PAGES:
                raise AnthropicCompatibilityError(
                    f"PDF contains more than {MAX_RENDERED_PAGES} scanned/image-only pages. "
                    "Split it into smaller parts before sending.",
                    status_code=413,
                )

            output.append(_text_block(f"{page_label}\n[Scanned page rendered as an image.]"))
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            output.append(_image_block(pixmap.tobytes("jpeg", jpg_quality=82)))
            rendered_pages += 1

        return output, {
            "code": source_kind,
            "pages": total_pages,
            "text_chars": extracted_chars,
            "rendered_pages": rendered_pages,
        }
    finally:
        document.close()


def _plain_text_document(
    block: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    source_type = source.get("type")
    media_type = str(source.get("media_type", "")).lower()
    if media_type != "text/plain" or source_type not in {"text", "base64"}:
        return None

    if source_type == "text":
        data = source.get("data")
        if not isinstance(data, str):
            raise AnthropicCompatibilityError("Plain-text document source.data must be a string.")
        text = data
    else:
        raw = _check_size(
            _decode_base64_data(source.get("data"), "Plain-text document"), "Document"
        )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AnthropicCompatibilityError(
                "Plain-text base64 document must use UTF-8 encoding."
            ) from exc

    title = _safe_title(block)
    if len(text) > MAX_EXTRACTED_CHARS:
        text = text[:MAX_EXTRACTED_CHARS] + "\n[Document text truncated.]"
    return [_text_block(f"[Document: {title}]\n{text}")], {
        "code": "document_text",
        "text_chars": len(text),
    }


def _readable_json(value: Any) -> str:
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
    if len(rendered) > MAX_HISTORY_BLOCK_CHARS:
        rendered = rendered[:MAX_HISTORY_BLOCK_CHARS] + "\n[Historical tool content truncated.]"
    return rendered


def _search_result_to_text(block: dict[str, Any]) -> str:
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

    header = f"[Search result: {title}]"
    if source:
        header += f"\nSource: {source}"
    return header + ("\n" + "\n".join(fragments) if fragments else "")


def _history_block_to_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type") or "unknown")
    if block_type == "search_result":
        return _search_result_to_text(block)
    if block_type == "server_tool_use":
        name = block.get("name") or "server_tool"
        return f"[Historical Anthropic server tool call: {name}]\n{_readable_json(block.get('input', {}))}"
    return f"[Historical Anthropic block converted to text: {block_type}]\n{_readable_json(block)}"


def _strip_ignored_block_fields(
    block: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    block = dict(block)
    if "cache_control" in block:
        block.pop("cache_control", None)
        events.append({"code": "cache_control_prefix_cache"})
    if "citations" in block:
        block.pop("citations", None)
        events.append({"code": "citations_unavailable"})
    return block


def _convert_content_blocks(blocks: list[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    converted: list[Any] = []
    events: list[dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict):
            converted.append(block)
            continue

        block = _strip_ignored_block_fields(block, events)
        if block.get("type") == "document":
            plain_text = _plain_text_document(block)
            replacement, document_stats = plain_text or convert_pdf_document(block)
            converted.extend(replacement)
            events.append(document_stats)
            continue

        block_type = block.get("type")
        if block_type in HISTORY_ONLY_CONTENT_TYPES:
            converted.append(_text_block(_history_block_to_text(block)))
            events.append({"code": "history_block_to_text", "block_type": block_type})
            continue

        # Anthropic tool results may themselves contain typed content blocks.
        if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
            nested, nested_events = _convert_content_blocks(block["content"])
            block = dict(block)
            block["content"] = nested
            events.extend(nested_events)

        if block_type not in VLLM_CONTENT_TYPES:
            raise AnthropicCompatibilityError(
                f"Anthropic content block type '{block_type}' is not supported by the local endpoint."
            )
        converted.append(block)

    return converted, events


def _compact_anthropic_image_history(
    payload: dict[str, Any], events: list[dict[str, Any]]
) -> None:
    """Keep only the newest raw images across an Anthropic conversation.

    Anthropic clients resend the complete message history on every request, so
    vLLM's per-prompt image limit includes historical images. Run this after
    document conversion so scanned PDF pages and tool-result images share the
    same budget.
    """
    image_locations: list[tuple[list[Any], int]] = []

    def collect(blocks: Any) -> None:
        if not isinstance(blocks, list):
            return
        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                image_locations.append((blocks, index))
            elif block.get("type") == "tool_result":
                collect(block.get("content"))

    for message in payload.get("messages", []):
        if isinstance(message, dict):
            collect(message.get("content"))

    omitted = max(0, len(image_locations) - MAX_PROMPT_IMAGES)
    if not omitted:
        return

    placeholder = (
        "[Earlier image omitted by the local gateway to stay within the "
        f"{MAX_PROMPT_IMAGES}-image prompt limit. Its visual data is not "
        "available in this turn.]"
    )
    for blocks, index in image_locations[:omitted]:
        blocks[index] = {"type": "text", "text": placeholder}

    events.append(
        {
            "code": "image_history_compacted",
            "images_seen": len(image_locations),
            "images_retained": len(image_locations) - omitted,
            "images_omitted": omitted,
        }
    )


def _apply_native_thinking(payload: dict[str, Any], events: list[dict[str, Any]]) -> None:
    if "thinking" not in payload:
        return
    thinking = payload.pop("thinking")
    if not isinstance(thinking, dict):
        raise AnthropicCompatibilityError("Anthropic thinking configuration must be an object.")
    thinking_type = thinking.get("type")
    if thinking_type in {"enabled", "adaptive"}:
        enabled = True
    elif thinking_type == "disabled":
        enabled = False
    else:
        raise AnthropicCompatibilityError(
            "Anthropic thinking.type must be 'enabled', 'adaptive', or 'disabled'."
        )

    kwargs = payload.get("chat_template_kwargs")
    if kwargs is None:
        kwargs = {}
        payload["chat_template_kwargs"] = kwargs
    if not isinstance(kwargs, dict):
        raise AnthropicCompatibilityError("chat_template_kwargs must be an object.")
    # The standard Anthropic field wins so clients can toggle thinking without
    # knowing about the vLLM-specific escape hatch.
    kwargs["enable_thinking"] = enabled
    events.append({"code": "thinking_enabled" if enabled else "thinking_disabled"})

    if thinking.get("display") == "omitted":
        events.append({"code": "thinking_display_not_filtered"})


def _normalize_tools(payload: dict[str, Any], events: list[dict[str, Any]]) -> None:
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
            raise AnthropicCompatibilityError(
                f"Anthropic server tool '{tool_type}' cannot run inside local vLLM. "
                "Expose it as a client tool in Hermes/the calling agent instead."
            )
        normalized.append(tool)
    payload["tools"] = normalized


def _event_header(events: list[dict[str, Any]]) -> str:
    codes: list[str] = []
    for event in events:
        code = str(event.get("code") or "")
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes)[:512]


def _route_to_local_model(payload: dict[str, Any], events: list[dict[str, Any]]) -> None:
    requested_model = payload.get("model")
    if requested_model != SERVED_MODEL:
        payload["model"] = SERVED_MODEL
        events.append(
            {
                "code": "model_routed_local",
                "requested_model": str(requested_model or "<default>"),
            }
        )


def _openai_file_document(part: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI file content part into our common document shape."""
    file_info = part.get("file") if isinstance(part.get("file"), dict) else part
    filename = str(file_info.get("filename") or part.get("filename") or "attached PDF")
    file_id = file_info.get("file_id") or part.get("file_id")
    if file_id:
        raise AnthropicCompatibilityError(
            "OpenAI file_id inputs require the Files API, which is not available on this "
            "local endpoint. Send file_data (base64/data URL) or file_url instead."
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
        raise AnthropicCompatibilityError(
            "OpenAI input_file/file content requires file_data, file_url, or file_id."
        )

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
            raise AnthropicCompatibilityError(
                "OpenAI file_data data URLs must contain valid base64 data."
            )
        media_type = match.group(1).lower()
        encoded = match.group(2)
    if not media_type:
        lowered = filename.lower()
        if lowered.endswith(".pdf"):
            media_type = "application/pdf"
        elif lowered.endswith((".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml")):
            media_type = "text/plain"
        else:
            # OpenAI examples allow raw base64 without a media type. PDF is the
            # useful and safest default because the parser validates its header.
            media_type = "application/pdf"

    if media_type not in {"application/pdf", "text/plain"}:
        raise AnthropicCompatibilityError(
            f"OpenAI file content type '{media_type}' is not supported locally. "
            "Use PDF/plain text, or send images as input_image/image_url blocks."
        )
    return {
        "type": "document",
        "title": filename,
        "source": {"type": "base64", "media_type": media_type, "data": encoded},
    }


def _openai_document_replacements(
    part: dict[str, Any], responses_style: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    document = _openai_file_document(part)
    plain_text = _plain_text_document(document)
    converted, stats = plain_text or convert_pdf_document(document)
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
        if responses_style:
            replacements.append({"type": "input_image", "image_url": data_url, "detail": "auto"})
        else:
            replacements.append(
                {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}}
            )
    stats = dict(stats)
    source_type = document.get("source", {}).get("media_type")
    stats["code"] = (
        "openai_input_file_pdf" if source_type == "application/pdf" else "openai_input_file_text"
    )
    return replacements, stats


def _convert_openai_content_parts(
    parts: list[Any], responses_style: bool
) -> tuple[list[Any], list[dict[str, Any]]]:
    converted: list[Any] = []
    events: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") in {"input_file", "file"}:
            replacements, stats = _openai_document_replacements(part, responses_style)
            converted.extend(replacements)
            events.append(stats)
        else:
            converted.append(part)
    return converted, events


def _convert_openai_documents(payload: dict[str, Any], events: list[dict[str, Any]]) -> None:
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            message["content"], part_events = _convert_openai_content_parts(
                message["content"], responses_style=False
            )
            events.extend(part_events)

    responses_input = payload.get("input")
    if not isinstance(responses_input, list):
        return
    converted_input: list[Any] = []
    for item in responses_input:
        if isinstance(item, dict) and item.get("type") in {"input_file", "file"}:
            replacements, stats = _openai_document_replacements(item, responses_style=True)
            converted_input.extend(replacements)
            events.append(stats)
            continue
        if isinstance(item, dict) and isinstance(item.get("content"), list):
            item = dict(item)
            item["content"], part_events = _convert_openai_content_parts(
                item["content"], responses_style=True
            )
            events.extend(part_events)
        converted_input.append(item)
    payload["input"] = converted_input


def transform_openai_request(payload: Any) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return payload, []
    events: list[dict[str, Any]] = []
    _route_to_local_model(payload, events)
    _convert_openai_documents(payload, events)

    # OpenAI-compatible coding clients commonly use reasoning_effort. vLLM
    # already understands it; explicitly align the Qwen chat template as well
    # so behavior is stable across chat-template upgrades.
    effort = payload.get("reasoning_effort")
    if isinstance(effort, str):
        kwargs = payload.get("chat_template_kwargs")
        if kwargs is None:
            kwargs = {}
            payload["chat_template_kwargs"] = kwargs
        if isinstance(kwargs, dict):
            kwargs["enable_thinking"] = effort.lower() not in {"none", "disabled"}
    return payload, events


def _normalize_proxy_path(path: str) -> str:
    aliases = {
        "models": "v1/models",
        "chat/completions": "v1/chat/completions",
        "completions": "v1/completions",
        "responses": "v1/responses",
        "messages": "v1/messages",
        "messages/count_tokens": "v1/messages/count_tokens",
        "openai/v1/models": "v1/models",
        "openai/v1/chat/completions": "v1/chat/completions",
        "openai/v1/completions": "v1/completions",
        "openai/v1/responses": "v1/responses",
    }
    if path in aliases:
        return aliases[path]

    azure_match = re.fullmatch(
        r"openai/deployments/[^/]+/(chat/completions|completions|responses)", path
    )
    if azure_match:
        return f"v1/{azure_match.group(1)}"
    return path


def transform_anthropic_request(payload: Any) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return payload, []

    events: list[dict[str, Any]] = []
    _route_to_local_model(payload, events)
    _apply_native_thinking(payload, events)

    if "cache_control" in payload:
        payload.pop("cache_control", None)
        events.append({"code": "cache_control_prefix_cache"})

    # Surface every unsupported future Anthropic request field instead of
    # letting Pydantic silently discard it.
    for key in list(payload):
        if key not in VLLM_ANTHROPIC_FIELDS:
            payload.pop(key, None)
            events.append({"code": f"ignored_field_{re.sub(r'[^a-zA-Z0-9_-]', '_', key)}"})

    _normalize_tools(payload, events)

    if isinstance(payload.get("system"), list):
        system, system_events = _convert_content_blocks(payload["system"])
        payload["system"] = system
        events.extend(system_events)

    for message in payload["messages"]:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        content, content_events = _convert_content_blocks(message["content"])
        message["content"] = content
        events.extend(content_events)
    _compact_anthropic_image_history(payload, events)
    return payload, events


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=settings.upstream_connect_timeout),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-local-anthropic-compat", "x-local-model-routed", "x-request-id"],
)
if settings.trusted_hosts != ("*",):
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.trusted_hosts))


def _provided_api_key(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return (
        request.headers.get("x-api-key", "").strip()
        or request.headers.get("api-key", "").strip()
        or request.headers.get("x-goog-api-key", "").strip()
        or request.query_params.get("key", "")
    )


def _key_is_valid(provided: str) -> bool:
    return bool(provided) and any(
        secrets.compare_digest(provided, expected) for expected in settings.api_keys
    )


@app.middleware("http")
async def gateway_controls(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            body_too_large = int(content_length) > settings.max_request_bytes
        except ValueError:
            body_too_large = True
        if body_too_large:
            return JSONResponse(
                status_code=413,
                content={"error": {"message": "Request body exceeds the configured limit."}},
                headers={"x-request-id": request_id},
            )

    public_path = request.url.path in {"/", "/healthz", "/readyz", "/v1/health"}
    if (
        settings.api_keys
        and request.method != "OPTIONS"
        and not public_path
        and not _key_is_valid(_provided_api_key(request))
    ):
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Missing or invalid gateway API key."}},
            headers={"www-authenticate": "Bearer", "x-request-id": request_id},
        )
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _ollama_details() -> dict[str, Any]:
    return {
        "parent_model": "",
        "format": settings.model_format,
        "family": settings.model_family,
        "families": [settings.model_family],
        "parameter_size": settings.model_parameter_size,
        "quantization_level": settings.model_quantization,
    }


def _ollama_model_entry(running: bool = False) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": SERVED_MODEL,
        "model": SERVED_MODEL,
        "modified_at": _utc_now(),
        "size": settings.model_size_bytes,
        "digest": f"local-{SERVED_MODEL}",
        "details": _ollama_details(),
    }
    if running:
        entry.update(
            {
                "expires_at": (datetime.now(UTC) + timedelta(days=3650))
                .isoformat()
                .replace("+00:00", "Z"),
                "size_vram": settings.model_vram_bytes,
                "context_length": MODEL_CONTEXT_LENGTH,
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


def _ollama_messages_to_openai(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise AnthropicCompatibilityError("Ollama messages must be an array.")
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
                function = call.get("function") if isinstance(call.get("function"), dict) else call
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


def _ollama_to_openai(payload: dict[str, Any], mode: str) -> tuple[dict[str, Any], str]:
    requested_model = str(payload.get("model") or SERVED_MODEL)
    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    if mode == "chat":
        messages = _ollama_messages_to_openai(payload.get("messages", []))
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
        messages.extend(_ollama_messages_to_openai([prompt_message]))

    request_payload: dict[str, Any] = {
        "model": SERVED_MODEL,
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


def _openai_tool_calls_to_ollama(tool_calls: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return converted
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
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


def _ollama_nonstream_response(
    data: dict[str, Any], requested_model: str, mode: str, started_ns: int
) -> dict[str, Any]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    tool_calls = _openai_tool_calls_to_ollama(message.get("tool_calls"))
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


async def _ollama_stream(
    upstream_response: httpx.Response,
    requested_model: str,
    mode: str,
    started_ns: int,
):
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
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") if isinstance(event.get("choices"), list) else []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])

            content = delta.get("content") or ""
            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                index = int(call.get("index") or 0)
                fragment = tool_fragments.setdefault(index, {"name": "", "arguments": ""})
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
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
        await upstream_response.aclose()


async def handle_ollama_request(request: Request, path: str, body: bytes):
    if request.method in {"GET", "HEAD"} and path == "api/version":
        return JSONResponse({"version": "0.6.0-local-gateway"})
    if request.method in {"GET", "HEAD"} and path == "api/tags":
        return JSONResponse({"models": [_ollama_model_entry()]})
    if request.method in {"GET", "HEAD"} and path == "api/ps":
        return JSONResponse({"models": [_ollama_model_entry(running=True)]})

    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "request body must be valid JSON"})
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400, content={"error": "request body must be a JSON object"}
        )

    if request.method == "POST" and path == "api/show":
        return JSONResponse(
            {
                "license": "Local model; see the model repository license.",
                "modelfile": f"FROM {SERVED_MODEL}",
                "parameters": f"num_ctx {MODEL_CONTEXT_LENGTH}",
                "template": "",
                "details": _ollama_details(),
                "model_info": {
                    "general.architecture": "qwen3",
                    "qwen3.context_length": MODEL_CONTEXT_LENGTH,
                },
                "capabilities": ["completion", "tools", "vision", "thinking"],
            }
        )
    if request.method == "POST" and path == "api/pull":
        return JSONResponse({"status": "success"})
    if request.method == "POST" and path in {"api/embed", "api/embeddings"}:
        return JSONResponse(
            status_code=400,
            content={
                "error": "This Qwen service is generative and does not provide embedding vectors."
            },
        )
    if request.method != "POST" or path not in {"api/chat", "api/generate"}:
        return JSONResponse(
            status_code=404, content={"error": f"unsupported Ollama endpoint: /{path}"}
        )

    mode = "chat" if path == "api/chat" else "generate"
    try:
        openai_payload, requested_model = _ollama_to_openai(payload, mode)
    except AnthropicCompatibilityError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    started_ns = time.perf_counter_ns()
    if not openai_payload["stream"]:
        try:
            upstream_response = await request.app.state.client.post(
                f"{UPSTREAM}/v1/chat/completions", json=openai_payload
            )
        except httpx.RequestError as exc:
            return JSONResponse(status_code=503, content={"error": f"vLLM unavailable: {exc}"})
        try:
            data = upstream_response.json()
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=upstream_response.status_code,
                content={"error": upstream_response.text[:2000]},
            )
        if upstream_response.status_code >= 400:
            error = data.get("error", data) if isinstance(data, dict) else data
            return JSONResponse(status_code=upstream_response.status_code, content={"error": error})
        return JSONResponse(_ollama_nonstream_response(data, requested_model, mode, started_ns))

    upstream_request = request.app.state.client.build_request(
        "POST", f"{UPSTREAM}/v1/chat/completions", json=openai_payload
    )
    try:
        upstream_response = await request.app.state.client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        return JSONResponse(status_code=503, content={"error": f"vLLM unavailable: {exc}"})
    if upstream_response.status_code >= 400:
        raw_error = await upstream_response.aread()
        await upstream_response.aclose()
        return JSONResponse(
            status_code=upstream_response.status_code,
            content={"error": raw_error.decode("utf-8", errors="replace")[:2000]},
        )
    return StreamingResponse(
        _ollama_stream(upstream_response, requested_model, mode, started_ns),
        media_type="application/x-ndjson",
    )


def _gemini_error(message: str, status_code: int = 400) -> JSONResponse:
    status = "INVALID_ARGUMENT" if status_code == 400 else "UNAVAILABLE"
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": status_code, "message": message, "status": status}},
    )


def _gemini_model() -> dict[str, Any]:
    return {
        "name": f"models/{SERVED_MODEL}",
        "baseModelId": SERVED_MODEL,
        "version": f"local-{settings.model_quantization.lower()}",
        "displayName": f"Local {SERVED_MODEL}",
        "description": "Local model served through vLLM Agent Gateway.",
        "inputTokenLimit": MODEL_CONTEXT_LENGTH,
        "outputTokenLimit": 32768,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent", "countTokens"],
    }


def _gemini_inline_part(part: dict[str, Any]) -> list[dict[str, Any]]:
    inline = part.get("inlineData") or part.get("inline_data")
    if not isinstance(inline, dict):
        return []
    media_type = str(inline.get("mimeType") or inline.get("mime_type") or "").lower()
    data = inline.get("data")
    if not isinstance(data, str):
        raise AnthropicCompatibilityError("Gemini inlineData.data must be a base64 string.")
    if media_type.startswith("image/"):
        return [{"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}]
    if media_type in {"application/pdf", "text/plain"}:
        document = {
            "type": "document",
            "title": str(part.get("displayName") or "Gemini inline document"),
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
        plain_text = _plain_text_document(document)
        converted, _ = plain_text or convert_pdf_document(document)
        output: list[dict[str, Any]] = []
        for block in converted:
            if block.get("type") == "text":
                output.append({"type": "text", "text": block.get("text", "")})
            elif block.get("type") == "image":
                source = block.get("source", {})
                output.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
                        },
                    }
                )
        return output
    raise AnthropicCompatibilityError(
        f"Gemini inlineData type '{media_type or '<missing>'}' is not supported by this model. "
        "Use images, PDF, or plain text; transcribe audio/video before sending."
    )


def _gemini_file_part(part: dict[str, Any]) -> list[dict[str, Any]]:
    file_data = part.get("fileData") or part.get("file_data")
    if not isinstance(file_data, dict):
        return []
    media_type = str(file_data.get("mimeType") or file_data.get("mime_type") or "").lower()
    uri = file_data.get("fileUri") or file_data.get("file_uri")
    if not isinstance(uri, str):
        raise AnthropicCompatibilityError("Gemini fileData.fileUri must be a URL.")
    if uri.startswith("gs://") or uri.startswith("files/"):
        raise AnthropicCompatibilityError(
            "Gemini Files/Google Cloud URIs require Google's Files API. Send inlineData or a public HTTPS URL."
        )
    if media_type.startswith("image/"):
        _validate_public_document_url(uri)
        return [{"type": "image_url", "image_url": {"url": uri}}]
    if media_type == "application/pdf":
        document = {
            "type": "document",
            "title": str(part.get("displayName") or "Gemini URL document"),
            "source": {"type": "url", "url": uri},
        }
        converted, _ = convert_pdf_document(document)
        output: list[dict[str, Any]] = []
        for block in converted:
            if block.get("type") == "text":
                output.append({"type": "text", "text": block.get("text", "")})
            else:
                source = block.get("source", {})
                output.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
                        },
                    }
                )
        return output
    raise AnthropicCompatibilityError(
        f"Gemini fileData type '{media_type or '<missing>'}' is not supported locally."
    )


def _gemini_to_openai(payload: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system_instruction = payload.get("systemInstruction") or payload.get("system_instruction")
    if isinstance(system_instruction, dict):
        system_text = "\n".join(
            str(part.get("text"))
            for part in system_instruction.get("parts", [])
            if isinstance(part, dict) and part.get("text") is not None
        )
        if system_text:
            messages.append({"role": "system", "content": system_text})

    tool_ids: dict[str, str] = {}
    contents = payload.get("contents")
    if not isinstance(contents, list):
        raise AnthropicCompatibilityError("Gemini contents must be an array.")
    for content_index, content in enumerate(contents):
        if not isinstance(content, dict):
            continue
        role = "assistant" if content.get("role") == "model" else "user"
        openai_parts: list[dict[str, Any]] = []
        function_calls: list[dict[str, Any]] = []
        function_responses: list[dict[str, Any]] = []
        for part_index, part in enumerate(content.get("parts") or []):
            if not isinstance(part, dict):
                continue
            if part.get("text") is not None:
                openai_parts.append({"type": "text", "text": str(part["text"])})
                continue
            openai_parts.extend(_gemini_inline_part(part))
            openai_parts.extend(_gemini_file_part(part))
            function_call = part.get("functionCall") or part.get("function_call")
            if isinstance(function_call, dict):
                name = str(function_call.get("name") or "tool")
                call_id = str(
                    function_call.get("id") or f"gemini-call-{content_index}-{part_index}"
                )
                tool_ids[name] = call_id
                function_calls.append(
                    {
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
                )
            function_response = part.get("functionResponse") or part.get("function_response")
            if isinstance(function_response, dict):
                function_responses.append(function_response)

        if openai_parts or function_calls:
            message: dict[str, Any] = {
                "role": role,
                "content": openai_parts if openai_parts else "",
            }
            if function_calls:
                message["tool_calls"] = function_calls
                message["role"] = "assistant"
            messages.append(message)
        for response_index, function_response in enumerate(function_responses):
            name = str(function_response.get("name") or "tool")
            response_value = function_response.get("response", {})
            response_text = (
                response_value
                if isinstance(response_value, str)
                else json.dumps(response_value, ensure_ascii=False, separators=(",", ":"))
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_ids.get(
                        name, f"gemini-result-{content_index}-{response_index}"
                    ),
                    "content": response_text,
                }
            )

    request_payload: dict[str, Any] = {
        "model": SERVED_MODEL,
        "messages": messages,
        "stream": False,
    }
    generation = payload.get("generationConfig") or payload.get("generation_config")
    if isinstance(generation, dict):
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
                request_payload[openai_key] = generation[gemini_key]
        mime = generation.get("responseMimeType")
        schema = generation.get("responseSchema") or generation.get("responseJsonSchema")
        if schema:
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "gemini_output", "schema": schema},
            }
        elif mime == "application/json":
            request_payload["response_format"] = {"type": "json_object"}
        thinking = generation.get("thinkingConfig") or generation.get("thinking_config")
        if isinstance(thinking, dict):
            budget = thinking.get("thinkingBudget", thinking.get("thinking_budget"))
            level = str(thinking.get("thinkingLevel") or thinking.get("thinking_level") or "")
            enabled = budget != 0 and level.lower() not in {"minimal", "none", "disabled"}
            request_payload["chat_template_kwargs"] = {"enable_thinking": enabled}
            request_payload["include_reasoning"] = enabled and bool(
                thinking.get("includeThoughts", thinking.get("include_thoughts", True))
            )

    tools: list[dict[str, Any]] = []
    for group in payload.get("tools") or []:
        if not isinstance(group, dict):
            continue
        declarations = group.get("functionDeclarations") or group.get("function_declarations") or []
        for declaration in declarations:
            if not isinstance(declaration, dict):
                continue
            function = {
                "name": str(declaration.get("name") or "tool"),
                "description": str(declaration.get("description") or ""),
                "parameters": declaration.get("parameters")
                or declaration.get("parametersJsonSchema")
                or {"type": "object", "properties": {}},
            }
            tools.append({"type": "function", "function": function})
    tool_config = payload.get("toolConfig") or payload.get("tool_config") or {}
    calling = (
        (
            tool_config.get("functionCallingConfig")
            or tool_config.get("function_calling_config")
            or {}
        )
        if isinstance(tool_config, dict)
        else {}
    )
    mode = str(calling.get("mode") or "AUTO").upper() if isinstance(calling, dict) else "AUTO"
    if tools and mode != "NONE":
        request_payload["tools"] = tools
        if mode == "ANY":
            allowed = (
                calling.get("allowedFunctionNames") or calling.get("allowed_function_names") or []
            )
            request_payload["tool_choice"] = (
                {"type": "function", "function": {"name": str(allowed[0])}}
                if len(allowed) == 1
                else "required"
            )
        else:
            request_payload["tool_choice"] = "auto"
    return request_payload


def _openai_to_gemini(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    parts: list[dict[str, Any]] = []
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        parts.append({"text": str(reasoning), "thought": True})
    if message.get("content"):
        parts.append({"text": str(message["content"])})
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
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
    return {
        "candidates": [
            {
                "content": {"parts": parts or [{"text": ""}], "role": "model"},
                "finishReason": finish_map.get(str(choice.get("finish_reason") or "stop"), "STOP"),
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": int(usage.get("prompt_tokens") or 0),
            "candidatesTokenCount": int(usage.get("completion_tokens") or 0),
            "totalTokenCount": int(usage.get("total_tokens") or 0),
        },
        "modelVersion": SERVED_MODEL,
        "responseId": str(data.get("id") or "local-gemini-response"),
    }


async def handle_gemini_request(request: Request, path: str, body: bytes):
    if request.method in {"GET", "HEAD"} and path == "v1beta/models":
        return JSONResponse({"models": [_gemini_model()]})
    match = re.fullmatch(
        r"v1beta/models/([^:]+)(?::(generateContent|streamGenerateContent|countTokens))?",
        path,
    )
    if not match:
        return _gemini_error(f"Unsupported Gemini endpoint: /{path}", 404)
    if request.method in {"GET", "HEAD"} and not match.group(2):
        return JSONResponse(_gemini_model())
    if request.method != "POST" or not match.group(2):
        return _gemini_error(f"Unsupported Gemini method for /{path}", 404)
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return _gemini_error("Request body must be valid JSON.")
    if not isinstance(payload, dict):
        return _gemini_error("Request body must be a JSON object.")
    try:
        openai_payload = await anyio.to_thread.run_sync(_gemini_to_openai, payload)
    except AnthropicCompatibilityError as exc:
        return _gemini_error(str(exc), exc.status_code)

    operation = match.group(2)
    if operation == "countTokens":
        try:
            tokenize_response = await request.app.state.client.post(
                f"{UPSTREAM}/tokenize",
                json={"model": SERVED_MODEL, "messages": openai_payload["messages"]},
            )
        except httpx.RequestError as exc:
            return _gemini_error(f"vLLM unavailable: {exc}", 503)
        if tokenize_response.status_code >= 400:
            return _gemini_error(tokenize_response.text[:2000], tokenize_response.status_code)
        token_data = tokenize_response.json()
        count = int(token_data.get("count") or len(token_data.get("tokens") or []))
        return JSONResponse({"totalTokens": count})

    try:
        upstream_response = await request.app.state.client.post(
            f"{UPSTREAM}/v1/chat/completions", json=openai_payload
        )
    except httpx.RequestError as exc:
        return _gemini_error(f"vLLM unavailable: {exc}", 503)
    try:
        upstream_data = upstream_response.json()
    except json.JSONDecodeError:
        return _gemini_error(upstream_response.text[:2000], upstream_response.status_code)
    if upstream_response.status_code >= 400:
        return _gemini_error(
            json.dumps(upstream_data, ensure_ascii=False)[:2000], upstream_response.status_code
        )
    result = _openai_to_gemini(upstream_data)
    if operation != "streamGenerateContent":
        return JSONResponse(result)
    if request.query_params.get("alt") == "sse":

        async def one_gemini_event():
            yield f"data: {json.dumps(result, ensure_ascii=False, separators=(',', ':'))}\r\n\r\n"

        return StreamingResponse(one_gemini_event(), media_type="text/event-stream")
    return JSONResponse([result])


async def proxy_request(request: Request, path: str):
    path = _normalize_proxy_path(path)

    if request.method in {"GET", "HEAD"} and not path:
        return JSONResponse(
            {
                "name": "vLLM Agent Gateway",
                "model": SERVED_MODEL,
                "upstream": "vLLM",
                "protocols": ["openai", "anthropic", "ollama", "gemini", "azure-openai"],
                "authentication": "api-key" if settings.api_keys else "disabled",
            }
        )

    if request.method in {"GET", "HEAD"} and path in {"healthz", "readyz", "v1/health"}:
        try:
            upstream_health = await request.app.state.client.get(f"{UPSTREAM}/health")
            ready = upstream_health.status_code == 200
        except httpx.RequestError:
            ready = False
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ok" if ready else "loading", "model": SERVED_MODEL},
        )

    body = await request.body()
    if path.startswith("api/"):
        return await handle_ollama_request(request, path, body)
    if path.startswith("v1beta/"):
        return await handle_gemini_request(request, path, body)

    compatibility_events: list[dict[str, Any]] = []

    if request.method == "POST" and path in {"v1/messages", "v1/messages/count_tokens"}:
        try:
            payload = json.loads(body)
            payload, compatibility_events = await anyio.to_thread.run_sync(
                transform_anthropic_request, payload
            )
            if compatibility_events:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                print(f"[anthropic-compat] transformations: {compatibility_events}", flush=True)
        except json.JSONDecodeError:
            pass  # Let the upstream API return its normal malformed-JSON response.
        except AnthropicCompatibilityError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": str(exc)},
                },
            )

    elif request.method == "POST" and path in {
        "v1/chat/completions",
        "v1/completions",
        "v1/responses",
    }:
        try:
            payload = json.loads(body)
            payload, compatibility_events = await anyio.to_thread.run_sync(
                transform_openai_request, payload
            )
            if compatibility_events:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                print(f"[openai-compat] transformations: {compatibility_events}", flush=True)
        except json.JSONDecodeError:
            pass
        except AnthropicCompatibilityError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {"type": "invalid_request_error", "message": str(exc)}},
            )

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    url = f"{UPSTREAM}/{path}" if path else f"{UPSTREAM}/"
    upstream_request = request.app.state.client.build_request(
        request.method,
        url,
        params=request.query_params,
        headers=headers,
        content=body,
    )

    try:
        upstream_response = await request.app.state.client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": f"vLLM upstream is unavailable: {exc}"},
            },
        )

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    compatibility_header = _event_header(compatibility_events)
    if compatibility_header:
        if path in {"v1/messages", "v1/messages/count_tokens"}:
            response_headers["x-local-anthropic-compat"] = compatibility_header
        if any(event.get("code") == "model_routed_local" for event in compatibility_events):
            response_headers["x-local-model-routed"] = SERVED_MODEL
    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream_response.aclose),
    )


@app.api_route("/", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_root(request: Request):
    return await proxy_request(request, "")


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_path(request: Request, path: str):
    return await proxy_request(request, path)
