"""ASGI entry point and v0.1 compatibility facade.

New integrations should import :func:`create_app`. The underscored helpers stay
available for users who imported the original single-file implementation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar
from urllib.parse import urlsplit

from .adapters import anthropic as anthropic_adapter
from .adapters import gemini as gemini_adapter
from .adapters import ollama as ollama_adapter
from .adapters import openai as openai_adapter
from .adapters.paths import normalize_proxy_path
from .application import create_app
from .config import settings
from .documents import (
    DocumentError,
    DocumentLimits,
    RemoteURLPolicy,
    validate_document_url,
)
from .documents import (
    convert_document as convert_document_async,
)
from .documents import (
    convert_pdf_document as convert_pdf_document_async,
)
from .errors import GatewayError

SERVED_MODEL = settings.served_model
MODEL_CONTEXT_LENGTH = settings.model_context_length
_T = TypeVar("_T")


class AnthropicCompatibilityError(GatewayError):
    """Deprecated name retained for source compatibility."""


def _run(coroutine: Coroutine[Any, Any, _T]) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    coroutine.close()
    raise RuntimeError("Legacy synchronous helpers cannot run inside an event loop; use adapters.")


def _limits() -> DocumentLimits:
    return DocumentLimits(
        max_raw_bytes=settings.max_pdf_bytes,
        max_pdf_pages=settings.max_pdf_pages,
        max_rendered_pages=settings.max_rendered_pages,
        max_extracted_chars=settings.max_extracted_chars,
        max_page_pixels=settings.pdf_max_page_pixels,
    )


async def _compat_convert(block: dict[str, Any]):
    blocks, stats = await convert_document_async(
        block,
        limits=_limits(),
        url_policy=RemoteURLPolicy(),
    )
    return blocks, stats.as_dict()


def _normalize_proxy_path(path: str) -> str:
    return normalize_proxy_path(path)


def _validate_public_document_url(url: str) -> None:
    hostname = urlsplit(url).hostname
    policy = RemoteURLPolicy(
        mode="allowlist",
        allowed_hosts=(hostname,) if hostname else (),
    )
    try:
        _run(validate_document_url(url, policy))
    except (DocumentError, GatewayError) as exc:
        raise AnthropicCompatibilityError(str(exc), exc.status_code) from exc


def convert_pdf_document(block: dict[str, Any]):
    try:
        blocks, stats = _run(
            convert_pdf_document_async(
                block,
                limits=_limits(),
                url_policy=RemoteURLPolicy(),
            )
        )
    except (DocumentError, GatewayError) as exc:
        raise AnthropicCompatibilityError(str(exc), exc.status_code) from exc
    return blocks, stats.as_dict()


def _convert_openai_documents(payload: dict[str, Any], events: list[dict[str, Any]]) -> None:
    try:
        _run(openai_adapter._convert_documents(payload, events, _compat_convert))
    except (DocumentError, GatewayError) as exc:
        raise AnthropicCompatibilityError(str(exc), exc.status_code) from exc


def transform_openai_request(payload: Any):
    try:
        return _run(
            openai_adapter.transform_request(
                payload,
                served_model=SERVED_MODEL,
                convert_document=_compat_convert,
            )
        )
    except (DocumentError, GatewayError) as exc:
        raise AnthropicCompatibilityError(str(exc), exc.status_code) from exc


def transform_anthropic_request(payload: Any):
    try:
        return _run(
            anthropic_adapter.transform_request(
                payload,
                served_model=SERVED_MODEL,
                convert_document=_compat_convert,
                max_prompt_images=settings.max_prompt_images,
            )
        )
    except (DocumentError, GatewayError) as exc:
        raise AnthropicCompatibilityError(str(exc), exc.status_code) from exc


def _ollama_to_openai(payload: dict[str, Any], mode: str):
    return ollama_adapter.to_openai(payload, mode, served_model=SERVED_MODEL)


def _ollama_nonstream_response(
    data: dict[str, Any], requested_model: str, mode: str, started_ns: int
):
    return ollama_adapter.nonstream_response(data, requested_model, mode, started_ns)


async def _unused_document(_block: dict[str, Any]):
    raise AnthropicCompatibilityError("Document conversion is unavailable in this helper.")


async def _validate_legacy_image_url(url: str) -> None:
    _validate_public_document_url(url)


def _gemini_to_openai(payload: dict[str, Any]):
    result, _events = _run(
        gemini_adapter.to_openai(
            payload,
            served_model=SERVED_MODEL,
            convert_document=_unused_document,
            validate_image_url=_validate_legacy_image_url,
        )
    )
    return result


def _openai_to_gemini(data: dict[str, Any]):
    return gemini_adapter.from_openai(data, served_model=SERVED_MODEL)


app = create_app(settings)

__all__ = [
    "AnthropicCompatibilityError",
    "MODEL_CONTEXT_LENGTH",
    "SERVED_MODEL",
    "app",
    "convert_pdf_document",
    "create_app",
    "settings",
    "transform_anthropic_request",
    "transform_openai_request",
]
