from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Collection
from contextlib import suppress
from dataclasses import dataclass

from starlette.types import ASGIApp, Receive, Scope, Send

from ._responses import send_json_error


@dataclass(frozen=True, slots=True)
class ConcurrencySnapshot:
    in_flight: int
    queued: int


class ConcurrencyRejected(Exception):
    def __init__(self, reason: str, retry_after: float) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


class ConcurrencyLimiter:
    """A fair single-event-loop limiter with a strictly bounded wait queue."""

    def __init__(
        self,
        *,
        max_in_flight: int,
        max_queue_size: int,
        queue_timeout: float,
    ) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be non-negative")
        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        self.max_in_flight = max_in_flight
        self.max_queue_size = max_queue_size
        self.queue_timeout = queue_timeout
        self._in_flight = 0
        self._waiters: deque[asyncio.Future[None]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self._in_flight < self.max_in_flight and not self._waiters:
                self._in_flight += 1
                return
            self._discard_done_waiters()
            if len(self._waiters) >= self.max_queue_size:
                raise ConcurrencyRejected("queue_full", self.queue_timeout)
            waiter = loop.create_future()
            self._waiters.append(waiter)

        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout=self.queue_timeout)
        except TimeoutError as exc:
            async with self._lock:
                if waiter.done():
                    return
                self._remove_waiter(waiter)
            raise ConcurrencyRejected("queue_timeout", self.queue_timeout) from exc
        except BaseException:
            async with self._lock:
                if waiter.done():
                    self._release_locked()
                else:
                    self._remove_waiter(waiter)
            raise

    async def release(self) -> None:
        async with self._lock:
            if self._in_flight < 1:
                raise RuntimeError("concurrency limiter released without an acquired slot")
            self._release_locked()

    async def snapshot(self) -> ConcurrencySnapshot:
        async with self._lock:
            self._discard_done_waiters()
            return ConcurrencySnapshot(self._in_flight, len(self._waiters))

    def _release_locked(self) -> None:
        self._in_flight -= 1
        self._discard_done_waiters()
        if self._waiters:
            waiter = self._waiters.popleft()
            self._in_flight += 1
            waiter.set_result(None)

    def _discard_done_waiters(self) -> None:
        while self._waiters and self._waiters[0].done():
            self._waiters.popleft()

    def _remove_waiter(self, target: asyncio.Future[None]) -> None:
        with suppress(ValueError):
            self._waiters.remove(target)


class ConcurrencyLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: ConcurrencyLimiter | None = None,
        max_in_flight: int = 4,
        max_queue_size: int = 16,
        queue_timeout: float = 30.0,
        excluded_paths: Collection[str] = (),
    ) -> None:
        self.app = app
        self.limiter = limiter or ConcurrencyLimiter(
            max_in_flight=max_in_flight,
            max_queue_size=max_queue_size,
            queue_timeout=queue_timeout,
        )
        self.excluded_paths = frozenset(excluded_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.excluded_paths:
            await self.app(scope, receive, send)
            return
        try:
            await self.limiter.acquire()
        except ConcurrencyRejected as exc:
            await send_json_error(
                send,
                status_code=429,
                message=(
                    "Gateway request queue is full."
                    if exc.reason == "queue_full"
                    else "Gateway request queue wait timed out."
                ),
                headers=((b"retry-after", str(max(1, math.ceil(exc.retry_after))).encode()),),
            )
            return
        try:
            await self.app(scope, receive, send)
        finally:
            await self.limiter.release()
