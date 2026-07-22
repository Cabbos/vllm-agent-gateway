from __future__ import annotations

import time
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .metrics import Counter, Histogram, MetricRegistry


def _protocol(path: str) -> str:
    if path.startswith("/api/"):
        return "ollama"
    if path.startswith("/v1beta/"):
        return "gemini"
    if path.startswith("/v1/messages"):
        return "anthropic"
    return "openai"


class RequestMetricsMiddleware:
    def __init__(self, app: ASGIApp, *, registry: MetricRegistry) -> None:
        self.app = app
        allowed = {
            "protocol": {"openai", "anthropic", "ollama", "gemini"},
            "outcome": {"success", "rejected", "upstream_error"},
        }
        self.requests: Counter = registry.counter(
            "gateway_requests_total",
            "Gateway HTTP requests.",
            label_names=("protocol", "outcome"),
            allowed_label_values=allowed,
        )
        self.duration: Histogram = registry.histogram(
            "gateway_request_duration_seconds",
            "Gateway request duration through the final streamed response byte.",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
            label_names=("protocol", "outcome"),
            allowed_label_values=allowed,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == "/gateway/metrics":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status_code = 500

        async def capture(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            protocol = _protocol(str(scope.get("path") or ""))
            outcome = (
                "success"
                if status_code < 400
                else "upstream_error"
                if status_code >= 500
                else "rejected"
            )
            labels: dict[str, Any] = {"protocol": protocol, "outcome": outcome}
            self.requests.inc(labels=labels)
            self.duration.observe(time.perf_counter() - started, labels=labels)
