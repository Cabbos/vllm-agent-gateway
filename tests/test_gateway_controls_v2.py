from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable

import pytest

from vllm_agent_gateway.middleware import (
    APIKeyAuthMiddleware,
    ConcurrencyLimiter,
    ConcurrencyLimitMiddleware,
    RateLimitMiddleware,
    RequestBodyLimitMiddleware,
    RequestIDMiddleware,
    TokenBucketRateLimiter,
    api_key_is_valid,
    extract_api_key,
)
from vllm_agent_gateway.observability import MetricRegistry, MetricSeriesLimitError


def _scope(
    *,
    method: str = "POST",
    path: str = "/v1/chat/completions",
    headers: Iterable[tuple[bytes, bytes]] = (),
    query_string: bytes = b"",
) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query_string,
        "headers": list(headers),
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
        "state": {},
    }


async def _invoke(app, scope: dict, messages: Iterable[dict] = ()) -> list[dict]:
    requests = iter([*messages, {"type": "http.disconnect"}])
    responses: list[dict] = []

    async def receive() -> dict:
        return next(requests)

    async def send(message: dict) -> None:
        responses.append(message)

    await app(scope, receive, send)
    return responses


async def _ok_app(scope, receive, send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    body = json.dumps(scope.get("state", {}), sort_keys=True).encode()
    await send({"type": "http.response.body", "body": body})


async def _read_body_app(scope, receive, send) -> None:
    body = bytearray()
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] != "http.request":
            break
        body.extend(message.get("body", b""))
        more_body = message.get("more_body", False)
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": bytes(body)})


def _status(messages: list[dict]) -> int:
    return next(
        message["status"] for message in messages if message["type"] == "http.response.start"
    )


def _headers(messages: list[dict]) -> dict[bytes, bytes]:
    start = next(message for message in messages if message["type"] == "http.response.start")
    return {name.lower(): value for name, value in start["headers"]}


@pytest.mark.parametrize(
    ("headers", "query", "expected", "source"),
    [
        ([(b"authorization", b"Bearer bearer-secret")], b"", "bearer-secret", "bearer"),
        ([(b"x-api-key", b"x-secret")], b"", "x-secret", "x-api-key"),
        ([(b"api-key", b"azure-secret")], b"", "azure-secret", "api-key"),
        ([(b"x-goog-api-key", b"google-secret")], b"", "google-secret", "x-goog-api-key"),
        ([], b"key=query-secret", "query-secret", "query"),
    ],
)
def test_extract_api_key_from_supported_locations(headers, query, expected, source):
    credential = extract_api_key(_scope(headers=headers, query_string=query))

    assert credential is not None
    assert credential.value == expected
    assert credential.source == source


def test_api_key_validation_compares_every_configured_key(monkeypatch):
    compared: list[str] = []

    def compare(provided: str, expected: str) -> bool:
        compared.append(expected)
        return provided == expected

    monkeypatch.setattr(
        "vllm_agent_gateway.middleware.authentication.secrets.compare_digest", compare
    )

    assert api_key_is_valid("first", ("first", "second", "third"))
    assert compared == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_auth_rejection_gets_request_id_and_does_not_run_app():
    called = False

    async def app(scope, receive, send) -> None:
        nonlocal called
        called = True

    wrapped = RequestIDMiddleware(
        APIKeyAuthMiddleware(app, api_keys=("secret",)), generator=lambda: "generated-id"
    )
    responses = await _invoke(wrapped, _scope(method="GET"))

    assert _status(responses) == 401
    assert _headers(responses)[b"www-authenticate"] == b"Bearer"
    assert _headers(responses)[b"x-request-id"] == b"generated-id"
    assert not called


@pytest.mark.asyncio
async def test_request_id_preserves_safe_input_and_replaces_invalid_input():
    wrapped = RequestIDMiddleware(_ok_app, generator=lambda: "replacement-id")

    safe = await _invoke(wrapped, _scope(method="GET", headers=[(b"x-request-id", b"caller-123")]))
    unsafe = await _invoke(
        wrapped,
        _scope(method="GET", headers=[(b"x-request-id", b"invalid id\r\nheader")]),
    )

    assert _headers(safe)[b"x-request-id"] == b"caller-123"
    assert _headers(unsafe)[b"x-request-id"] == b"replacement-id"


@pytest.mark.asyncio
async def test_body_limit_counts_chunked_receive_without_content_length():
    wrapped = RequestBodyLimitMiddleware(_read_body_app, max_bytes=5)
    responses = await _invoke(
        wrapped,
        _scope(),
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": False},
        ],
    )

    assert _status(responses) == 413
    assert b"configured limit" in responses[-1]["body"]


@pytest.mark.asyncio
async def test_body_limit_allows_multiple_chunks_up_to_limit():
    wrapped = RequestBodyLimitMiddleware(_read_body_app, max_bytes=6)
    responses = await _invoke(
        wrapped,
        _scope(),
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": False},
        ],
    )

    assert _status(responses) == 200
    assert responses[-1]["body"] == b"123456"


@pytest.mark.asyncio
async def test_body_limit_rejects_oversized_content_length_before_app():
    called = False

    async def app(scope, receive, send) -> None:
        nonlocal called
        called = True

    wrapped = RequestBodyLimitMiddleware(app, max_bytes=5)
    responses = await _invoke(wrapped, _scope(headers=[(b"content-length", b"6")]))

    assert _status(responses) == 413
    assert not called


@pytest.mark.asyncio
async def test_concurrency_middleware_bounds_queue_and_returns_retry_after():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def blocking_app(scope, receive, send) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    limiter = ConcurrencyLimiter(max_in_flight=1, max_queue_size=1, queue_timeout=5)
    wrapped = ConcurrencyLimitMiddleware(blocking_app, limiter=limiter)
    first = asyncio.create_task(_invoke(wrapped, _scope()))
    await entered.wait()
    second = asyncio.create_task(_invoke(wrapped, _scope()))
    for _ in range(10):
        if (await limiter.snapshot()).queued == 1:
            break
        await asyncio.sleep(0)

    rejected = await _invoke(wrapped, _scope())
    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert _status(rejected) == 429
    assert _headers(rejected)[b"retry-after"] == b"5"
    assert _status(first_result) == 200
    assert _status(second_result) == 200
    assert calls == 2


@pytest.mark.asyncio
async def test_token_bucket_rate_limit_is_isolated_per_api_key():
    now = [100.0]
    limiter = TokenBucketRateLimiter(
        requests_per_minute=60,
        burst=1,
        clock=lambda: now[0],
    )
    wrapped = APIKeyAuthMiddleware(
        RateLimitMiddleware(_ok_app, limiter=limiter), api_keys=("alpha", "beta")
    )

    alpha_first = await _invoke(wrapped, _scope(headers=[(b"x-api-key", b"alpha")]))
    alpha_second = await _invoke(wrapped, _scope(headers=[(b"x-api-key", b"alpha")]))
    beta_first = await _invoke(wrapped, _scope(headers=[(b"x-api-key", b"beta")]))

    assert _status(alpha_first) == 200
    assert _status(alpha_second) == 429
    assert _headers(alpha_second)[b"retry-after"] == b"1"
    assert _status(beta_first) == 200
    assert "alpha" not in limiter._buckets
    assert "beta" not in limiter._buckets


def test_dependency_free_metrics_exposition_and_cardinality_guards():
    registry = MetricRegistry()
    requests = registry.counter(
        "gateway_requests_total",
        "Gateway requests.",
        label_names=("protocol", "outcome"),
        allowed_label_values={
            "protocol": {"openai", "anthropic"},
            "outcome": {"success", "rejected"},
        },
        max_series=2,
    )
    latency = registry.histogram(
        "gateway_request_duration_seconds",
        "Gateway request duration.",
        buckets=(0.1, 1.0),
    )
    requests.inc(labels={"protocol": "openai", "outcome": "success"})
    requests.inc(labels={"protocol": "anthropic", "outcome": "rejected"})
    latency.observe(0.25)

    output = registry.render()

    assert 'gateway_requests_total{protocol="openai",outcome="success"} 1' in output
    assert 'gateway_request_duration_seconds_bucket{le="1"} 1' in output
    assert "gateway_request_duration_seconds_count 1" in output
    with pytest.raises(MetricSeriesLimitError):
        requests.inc(labels={"protocol": "anthropic", "outcome": "success"})
    with pytest.raises(ValueError, match="forbidden"):
        registry.counter("unsafe_total", "Unsafe.", label_names=("api_key",))
