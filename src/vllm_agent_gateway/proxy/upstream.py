from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

from .streaming import UpstreamStreamingResponse, close_response_shielded

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
CLIENT_CREDENTIAL_HEADERS = {"authorization", "x-api-key", "api-key", "x-goog-api-key"}
logger = logging.getLogger(__name__)


def forwarded_headers(
    headers: Mapping[str, str], *, strip_credentials: bool = False
) -> dict[str, str]:
    excluded = HOP_BY_HOP_HEADERS | (CLIENT_CREDENTIAL_HEADERS if strip_credentials else set())
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


async def iter_and_close(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        if response.is_stream_consumed:
            if response.content:
                yield response.content
            return
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await close_response_shielded(response)


async def forward_streaming(
    *,
    client: httpx.AsyncClient,
    request: Request,
    upstream_url: str,
    body: bytes,
    query_params: Sequence[tuple[str, str]] = (),
    upstream_api_key: str = "",
    response_headers: Mapping[str, str] | None = None,
):
    headers = forwarded_headers(request.headers, strip_credentials=True)
    if upstream_api_key:
        headers["authorization"] = f"Bearer {upstream_api_key}"
    upstream_request = client.build_request(
        request.method,
        upstream_url,
        params=query_params,
        headers=headers,
        content=body,
    )
    try:
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        logger.warning("vLLM upstream request failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content={"error": {"type": "api_error", "message": "vLLM upstream is unavailable."}},
        )

    headers = forwarded_headers(upstream_response.headers)
    if response_headers:
        headers.update(response_headers)
    return UpstreamStreamingResponse(
        iter_and_close(upstream_response),
        upstream_response=upstream_response,
        status_code=upstream_response.status_code,
        headers=headers,
    )
