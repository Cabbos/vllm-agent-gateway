from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .adapters import anthropic, gemini, openai
from .adapters import ollama as ollama_adapter
from .adapters.common import event_codes
from .adapters.gemini_stream import convert_openai_sse
from .adapters.paths import normalize_proxy_path
from .config import Settings, settings
from .document_service import DocumentService
from .documents import DocumentError, DocumentLimits, RemoteURLPolicy
from .errors import GatewayError
from .middleware import (
    APIKeyAuthMiddleware,
    ConcurrencyLimitMiddleware,
    RateLimitMiddleware,
    RequestBodyLimitMiddleware,
    RequestIDMiddleware,
    TokenBucketRateLimiter,
    rate_limit_identity,
)
from .observability import MetricRegistry, RequestMetricsMiddleware
from .proxy.streaming import UpstreamStreamingResponse, close_response_shielded
from .proxy.upstream import forward_streaming

logger = logging.getLogger(__name__)


def _anonymous_rate_identity(_scope: Any) -> str:
    return "anonymous"


def _query_without_credentials(request: Request) -> list[tuple[str, str]]:
    return [
        (key, value)
        for key, value in request.query_params.multi_items()
        if key not in {"key", "api-version"}
    ]


def _upstream_headers(config: Settings) -> dict[str, str]:
    return {"authorization": f"Bearer {config.upstream_api_key}"} if config.upstream_api_key else {}


def _upstream_unavailable(protocol: str = "openai") -> JSONResponse:
    if protocol == "ollama":
        return JSONResponse(status_code=503, content={"error": "vLLM upstream is unavailable."})
    if protocol == "gemini":
        return _gemini_error("vLLM upstream is unavailable.", 503)
    return JSONResponse(
        status_code=503,
        content={"error": {"type": "api_error", "message": "vLLM upstream is unavailable."}},
    )


def _gemini_error(message: str, status_code: int = 400) -> JSONResponse:
    status = "INVALID_ARGUMENT" if status_code < 500 else "UNAVAILABLE"
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": status_code, "message": message, "status": status}},
    )


def _document_service(config: Settings) -> DocumentService:
    return DocumentService(
        limits=DocumentLimits(
            max_raw_bytes=config.max_pdf_bytes,
            max_pdf_pages=config.max_pdf_pages,
            max_rendered_pages=config.max_rendered_pages,
            max_extracted_chars=config.max_extracted_chars,
            max_page_pixels=config.pdf_max_page_pixels,
        ),
        url_policy=RemoteURLPolicy(
            mode=config.document_url_policy,
            allowed_hosts=config.document_allowed_hosts,
            extra_allowed_networks=config.document_extra_allowed_networks,
        ),
        concurrency=config.pdf_conversion_concurrency,
        timeout_seconds=config.pdf_conversion_timeout_seconds,
    )


async def _send_openai_stream(
    request: Request,
    payload: dict[str, Any],
    config: Settings,
) -> httpx.Response | JSONResponse:
    client: httpx.AsyncClient = request.app.state.client
    upstream_request = client.build_request(
        "POST",
        f"{config.upstream}/v1/chat/completions",
        json=payload,
        headers=_upstream_headers(config),
    )
    try:
        return await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        logger.warning("vLLM streaming request failed: %s", type(exc).__name__)
        return _upstream_unavailable()


async def _ollama_request(request: Request, path: str, body: bytes, config: Settings) -> Response:
    details = ollama_adapter.model_details(
        model_format=config.model_format,
        model_family=config.model_family,
        parameter_size=config.model_parameter_size,
        quantization=config.model_quantization,
    )
    entry_args = {
        "served_model": config.served_model,
        "details": details,
        "size_bytes": config.model_size_bytes,
        "vram_bytes": config.model_vram_bytes,
        "context_length": config.model_context_length,
    }
    if request.method in {"GET", "HEAD"} and path == "api/version":
        return JSONResponse({"version": "0.6.0-local-gateway"})
    if request.method in {"GET", "HEAD"} and path == "api/tags":
        return JSONResponse({"models": [ollama_adapter.model_entry(**entry_args)]})
    if request.method in {"GET", "HEAD"} and path == "api/ps":
        return JSONResponse({"models": [ollama_adapter.model_entry(**entry_args, running=True)]})

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
                "modelfile": f"FROM {config.served_model}",
                "parameters": f"num_ctx {config.model_context_length}",
                "template": "",
                "details": details,
                "model_info": {
                    "general.architecture": config.model_family,
                    f"{config.model_family}.context_length": config.model_context_length,
                },
                "capabilities": ["completion", "tools", "vision", "thinking"],
            }
        )
    if request.method == "POST" and path == "api/pull":
        return JSONResponse({"status": "success"})
    if request.method == "POST" and path in {"api/embed", "api/embeddings"}:
        return JSONResponse(
            status_code=400,
            content={"error": "This service does not provide embedding vectors."},
        )
    if request.method != "POST" or path not in {"api/chat", "api/generate"}:
        return JSONResponse(
            status_code=404,
            content={"error": f"unsupported Ollama endpoint: /{path}"},
        )

    mode = "chat" if path == "api/chat" else "generate"
    try:
        openai_payload, requested_model = ollama_adapter.to_openai(
            payload, mode, served_model=config.served_model
        )
    except GatewayError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})
    client: httpx.AsyncClient = request.app.state.client
    started_ns = time.perf_counter_ns()
    if not openai_payload["stream"]:
        try:
            upstream_response = await client.post(
                f"{config.upstream}/v1/chat/completions",
                json=openai_payload,
                headers=_upstream_headers(config),
            )
        except httpx.RequestError as exc:
            logger.warning("vLLM Ollama request failed: %s", type(exc).__name__)
            return _upstream_unavailable("ollama")
        try:
            data = upstream_response.json()
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=upstream_response.status_code,
                content={"error": "vLLM returned a non-JSON response."},
            )
        if upstream_response.status_code >= 400:
            error = data.get("error", data) if isinstance(data, dict) else data
            return JSONResponse(status_code=upstream_response.status_code, content={"error": error})
        return JSONResponse(
            ollama_adapter.nonstream_response(data, requested_model, mode, started_ns)
        )

    upstream = await _send_openai_stream(request, openai_payload, config)
    if isinstance(upstream, JSONResponse):
        return _upstream_unavailable("ollama")
    if upstream.status_code >= 400:
        await upstream.aread()
        await close_response_shielded(upstream)
        return JSONResponse(
            status_code=upstream.status_code,
            content={"error": "vLLM rejected the request."},
        )
    return UpstreamStreamingResponse(
        ollama_adapter.stream_response(upstream, requested_model, mode, started_ns),
        upstream_response=upstream,
        media_type="application/x-ndjson",
    )


async def _close_after_gemini_stream(
    upstream: httpx.Response,
    *,
    framing: str,
    model_version: str,
):
    try:
        async for chunk in convert_openai_sse(
            upstream.aiter_bytes(),
            framing=framing,
            model_version=model_version,
        ):
            yield chunk
    finally:
        await close_response_shielded(upstream)


async def _gemini_request(request: Request, path: str, body: bytes, config: Settings) -> Response:
    model = gemini.model_metadata(
        served_model=config.served_model,
        context_length=config.model_context_length,
        quantization=config.model_quantization,
    )
    if request.method in {"GET", "HEAD"} and path == "v1beta/models":
        return JSONResponse({"models": [model]})
    match = re.fullmatch(
        r"v1beta/models/([^:]+)(?::(generateContent|streamGenerateContent|countTokens))?",
        path,
    )
    if not match:
        return _gemini_error(f"Unsupported Gemini endpoint: /{path}", 404)
    if request.method in {"GET", "HEAD"} and not match.group(2):
        return JSONResponse(model)
    if request.method != "POST" or not match.group(2):
        return _gemini_error(f"Unsupported Gemini method for /{path}", 404)
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return _gemini_error("Request body must be valid JSON.")
    if not isinstance(payload, dict):
        return _gemini_error("Request body must be a JSON object.")

    service: DocumentService = request.app.state.document_service
    try:
        openai_payload, _events = await gemini.to_openai(
            payload,
            served_model=config.served_model,
            convert_document=service.convert,
            validate_image_url=service.validate_image_url,
        )
    except (GatewayError, DocumentError) as exc:
        return _gemini_error(str(exc), getattr(exc, "status_code", 400))

    client: httpx.AsyncClient = request.app.state.client
    operation = match.group(2)
    if operation == "countTokens":
        try:
            response = await client.post(
                f"{config.upstream}/tokenize",
                json={"model": config.served_model, "messages": openai_payload["messages"]},
                headers=_upstream_headers(config),
            )
        except httpx.RequestError as exc:
            logger.warning("vLLM tokenize request failed: %s", type(exc).__name__)
            return _upstream_unavailable("gemini")
        if response.status_code >= 400:
            return _gemini_error("vLLM rejected the token-count request.", response.status_code)
        try:
            token_data = response.json()
        except json.JSONDecodeError:
            return _gemini_error("vLLM returned a non-JSON token response.", 502)
        count = int(token_data.get("count") or len(token_data.get("tokens") or []))
        return JSONResponse({"totalTokens": count})

    if operation == "streamGenerateContent":
        openai_payload["stream"] = True
        openai_payload["stream_options"] = {"include_usage": True}
        upstream = await _send_openai_stream(request, openai_payload, config)
        if isinstance(upstream, JSONResponse):
            return _upstream_unavailable("gemini")
        if upstream.status_code >= 400:
            await upstream.aread()
            await close_response_shielded(upstream)
            return _gemini_error("vLLM rejected the request.", upstream.status_code)
        framing = "sse" if request.query_params.get("alt") == "sse" else "json-array"
        media_type = "text/event-stream" if framing == "sse" else "application/json"
        return UpstreamStreamingResponse(
            _close_after_gemini_stream(
                upstream,
                framing=framing,
                model_version=config.served_model,
            ),
            upstream_response=upstream,
            media_type=media_type,
        )

    try:
        response = await client.post(
            f"{config.upstream}/v1/chat/completions",
            json=openai_payload,
            headers=_upstream_headers(config),
        )
    except httpx.RequestError as exc:
        logger.warning("vLLM Gemini request failed: %s", type(exc).__name__)
        return _upstream_unavailable("gemini")
    try:
        data = response.json()
    except json.JSONDecodeError:
        return _gemini_error("vLLM returned a non-JSON response.", 502)
    if response.status_code >= 400:
        return _gemini_error("vLLM rejected the request.", response.status_code)
    return JSONResponse(gemini.from_openai(data, served_model=config.served_model))


async def _proxy_request(request: Request, path: str, config: Settings) -> Response:
    path = normalize_proxy_path(path)
    if path.rstrip("/") == "gateway/metrics":
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Gateway metrics are disabled or unavailable."}},
        )
    if request.method in {"GET", "HEAD"} and not path:
        return JSONResponse(
            {
                "name": "vLLM Agent Gateway",
                "version": "0.2.0",
                "model": config.served_model,
                "upstream": "vLLM",
                "protocols": ["openai", "anthropic", "ollama", "gemini", "azure-openai"],
                "authentication": "api-key" if config.api_keys else "disabled",
            }
        )
    if request.method in {"GET", "HEAD"} and path in {"healthz", "readyz", "v1/health"}:
        try:
            response = await request.app.state.client.get(
                f"{config.upstream}/health", headers=_upstream_headers(config)
            )
            ready = response.status_code == 200
        except httpx.RequestError:
            ready = False
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ok" if ready else "loading", "model": config.served_model},
        )

    body = await request.body()
    if path.startswith("api/"):
        return await _ollama_request(request, path, body, config)
    if path.startswith("v1beta/"):
        return await _gemini_request(request, path, body, config)

    compatibility_events: list[dict[str, Any]] = []
    service: DocumentService = request.app.state.document_service
    try:
        if request.method == "POST" and path in {"v1/messages", "v1/messages/count_tokens"}:
            payload = json.loads(body)
            payload, compatibility_events = await anthropic.transform_request(
                payload,
                served_model=config.served_model,
                convert_document=service.convert,
                max_prompt_images=config.max_prompt_images,
            )
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        elif request.method == "POST" and path in {
            "v1/chat/completions",
            "v1/completions",
            "v1/responses",
        }:
            payload = json.loads(body)
            payload, compatibility_events = await openai.transform_request(
                payload,
                served_model=config.served_model,
                convert_document=service.convert,
            )
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    except json.JSONDecodeError:
        pass
    except (GatewayError, DocumentError) as exc:
        status_code = getattr(exc, "status_code", 400)
        if path.startswith("v1/messages"):
            return JSONResponse(
                status_code=status_code,
                content={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": str(exc)},
                },
            )
        return JSONResponse(
            status_code=status_code,
            content={"error": {"type": "invalid_request_error", "message": str(exc)}},
        )

    extra_headers: dict[str, str] = {}
    codes = event_codes(compatibility_events)
    if codes and path.startswith("v1/messages"):
        extra_headers["x-local-anthropic-compat"] = codes
    if any(event.get("code") == "model_routed_local" for event in compatibility_events):
        extra_headers["x-local-model-routed"] = config.served_model
    return await forward_streaming(
        client=request.app.state.client,
        request=request,
        upstream_url=f"{config.upstream}/{path}" if path else f"{config.upstream}/",
        body=body,
        query_params=_query_without_credentials(request),
        upstream_api_key=config.upstream_api_key,
        response_headers=extra_headers,
    )


def create_app(
    config: Settings = settings,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    registry = MetricRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=config.upstream_connect_timeout),
            limits=httpx.Limits(
                max_connections=config.upstream_max_connections,
                max_keepalive_connections=config.upstream_max_keepalive_connections,
            ),
            transport=upstream_transport,
        )
        app.state.document_service = _document_service(config)
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.settings = config
    app.state.metrics_registry = registry

    if config.metrics_enabled:

        @app.api_route("/gateway/metrics", methods=["GET", "HEAD"])
        async def gateway_metrics(request: Request) -> Response:
            rendered = registry.render()
            return Response(
                content="" if request.method == "HEAD" else rendered,
                media_type="text/plain; version=0.0.4",
            )

    @app.api_route("/", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy_root(request: Request):
        return await _proxy_request(request, "", config)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy_path(request: Request, path: str):
        return await _proxy_request(request, path, config)

    if config.trusted_hosts != ("*",):
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(config.trusted_hosts))
    inexpensive_paths = {"/", "/healthz", "/readyz", "/v1/health", "/gateway/metrics"}
    if config.max_inflight > 0:
        app.add_middleware(
            ConcurrencyLimitMiddleware,
            max_in_flight=config.max_inflight,
            max_queue_size=config.max_queue_size,
            queue_timeout=config.queue_timeout_seconds,
            excluded_paths=inexpensive_paths,
        )
    if config.requests_per_minute > 0:
        identity_resolver = rate_limit_identity if config.api_keys else _anonymous_rate_identity
        app.add_middleware(
            RateLimitMiddleware,
            limiter=TokenBucketRateLimiter(
                requests_per_minute=config.requests_per_minute,
                burst=config.rate_limit_burst,
            ),
            identity_resolver=identity_resolver,
            excluded_paths=inexpensive_paths,
        )
    app.add_middleware(
        APIKeyAuthMiddleware,
        api_keys=config.api_keys,
        public_paths={"/", "/healthz", "/readyz", "/v1/health"},
    )
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=config.max_request_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "x-local-anthropic-compat",
            "x-local-model-routed",
            "x-request-id",
        ],
    )
    if config.metrics_enabled:
        app.add_middleware(RequestMetricsMiddleware, registry=registry)
    app.add_middleware(RequestIDMiddleware)
    return app
