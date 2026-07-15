from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ._responses import send_json_error


class RequestBodyTooLarge(Exception):
    pass


def _content_length_exceeds_limit(scope: Scope, limit: int) -> bool:
    values = [
        value for name, value in scope.get("headers", []) if name.lower() == b"content-length"
    ]
    if not values:
        return False
    try:
        lengths = [int(value.decode("ascii")) for value in values]
    except (UnicodeDecodeError, ValueError):
        return True
    return any(length < 0 or length > limit for length in lengths) or len(set(lengths)) != 1


class RequestBodyLimitMiddleware:
    """Enforce a request-body limit from both headers and actual ASGI chunks."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if _content_length_exceeds_limit(scope, self.max_bytes):
            await self._reject(send)
            return

        received = 0
        rejected = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    raise RequestBodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if rejected:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not rejected or response_started:
                raise
        if rejected:
            if response_started:
                raise RequestBodyTooLarge
            await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        await send_json_error(
            send,
            status_code=413,
            message="Request body exceeds the configured limit.",
        )
