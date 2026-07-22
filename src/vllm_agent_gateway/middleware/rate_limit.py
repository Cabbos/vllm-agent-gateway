from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Collection
from dataclasses import dataclass

from starlette.types import ASGIApp, Receive, Scope, Send

from ._responses import send_json_error
from .authentication import api_key_fingerprint, extract_api_key_value


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after: float


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    def __init__(
        self,
        *,
        requests_per_minute: float,
        burst: int,
        max_buckets: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if burst < 1:
            raise ValueError("burst must be at least 1")
        if max_buckets < 1:
            raise ValueError("max_buckets must be at least 1")
        self.requests_per_minute = float(requests_per_minute)
        self.burst = burst
        self.max_buckets = max_buckets
        self.clock = clock
        self._refill_per_second = self.requests_per_minute / 60.0
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, identity: str, *, cost: float = 1.0) -> RateLimitDecision:
        """Consume tokens for an opaque identity; callers must not pass it to metrics/logs."""
        if not identity:
            raise ValueError("identity must not be empty")
        if cost <= 0 or cost > self.burst:
            raise ValueError("cost must be positive and no greater than burst")

        now = self.clock()
        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                if len(self._buckets) >= self.max_buckets:
                    self._buckets.popitem(last=False)
                bucket = _Bucket(float(self.burst), now)
                self._buckets[identity] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(
                    float(self.burst), bucket.tokens + elapsed * self._refill_per_second
                )
                bucket.updated_at = now
                self._buckets.move_to_end(identity)

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return RateLimitDecision(True, math.floor(bucket.tokens), 0.0)
            retry_after = (cost - bucket.tokens) / self._refill_per_second
            return RateLimitDecision(False, 0, retry_after)

    def check_api_key(self, api_key: str, *, cost: float = 1.0) -> RateLimitDecision:
        return self.check(api_key_fingerprint(api_key), cost=cost)


def rate_limit_identity(scope: Scope) -> str:
    state = scope.get("state", {})
    authenticated_id = state.get("gateway_api_key_id")
    if isinstance(authenticated_id, str) and authenticated_id:
        return authenticated_id
    provided = extract_api_key_value(scope)
    return api_key_fingerprint(provided) if provided else "anonymous"


class RateLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: TokenBucketRateLimiter,
        identity_resolver: Callable[[Scope], str] = rate_limit_identity,
        excluded_paths: Collection[str] = (),
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.identity_resolver = identity_resolver
        self.excluded_paths = frozenset(excluded_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.excluded_paths:
            await self.app(scope, receive, send)
            return
        decision = self.limiter.check(self.identity_resolver(scope))
        if not decision.allowed:
            await send_json_error(
                send,
                status_code=429,
                message="Gateway API key rate limit exceeded.",
                headers=((b"retry-after", str(max(1, math.ceil(decision.retry_after))).encode()),),
            )
            return
        await self.app(scope, receive, send)
