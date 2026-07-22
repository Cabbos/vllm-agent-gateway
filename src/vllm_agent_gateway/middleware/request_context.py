from __future__ import annotations

import re
import uuid
from collections.abc import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def request_id_from_scope(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name.lower() == b"x-request-id":
            candidate = value.decode("latin-1").strip()
            return candidate if _REQUEST_ID_PATTERN.fullmatch(candidate) else None
    return None


class RequestIDMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        generator: Callable[[], str] | None = None,
    ) -> None:
        self.app = app
        self.generator = generator or (lambda: str(uuid.uuid4()))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = request_id_from_scope(scope) or self.generator()
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_request_id)
