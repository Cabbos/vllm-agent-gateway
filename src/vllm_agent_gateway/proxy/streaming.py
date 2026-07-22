from __future__ import annotations

from typing import Any

import anyio
import httpx
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


async def close_response_shielded(response: httpx.Response) -> None:
    """Finish closing an upstream socket even inside a cancelled request scope."""

    with anyio.CancelScope(shield=True):
        await response.aclose()


class UpstreamStreamingResponse(StreamingResponse):
    """A streaming response whose upstream is closed on every ASGI exit path."""

    def __init__(
        self,
        content: Any,
        *,
        upstream_response: httpx.Response,
        **kwargs: Any,
    ) -> None:
        super().__init__(content, **kwargs)
        self.upstream_response = upstream_response

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await close_response_shielded(self.upstream_response)
