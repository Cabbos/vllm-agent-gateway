from __future__ import annotations

from typing import Any

import anyio

from .documents import (
    DocumentLimits,
    RemoteURLPolicy,
    convert_loaded_pdf,
    convert_plain_document,
    load_document_source,
    validate_document_url,
)
from .errors import GatewayError


class DocumentService:
    """Applies URL policy and resource limits around document conversion."""

    def __init__(
        self,
        *,
        limits: DocumentLimits,
        url_policy: RemoteURLPolicy,
        concurrency: int,
        timeout_seconds: float,
    ) -> None:
        self.limits = limits
        self.url_policy = url_policy
        self.timeout_seconds = timeout_seconds
        self._thread_limiter = anyio.CapacityLimiter(max(1, concurrency))

    async def convert(self, block: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            with anyio.fail_after(self.timeout_seconds):
                loaded = await load_document_source(
                    block,
                    limits=self.limits,
                    url_policy=self.url_policy,
                )
                converter = (
                    convert_loaded_pdf
                    if loaded.media_type == "application/pdf"
                    else convert_plain_document
                )
                blocks, stats = await anyio.to_thread.run_sync(
                    lambda: converter(loaded, limits=self.limits),
                    abandon_on_cancel=True,
                    limiter=self._thread_limiter,
                )
                return blocks, stats.as_dict()
        except TimeoutError as exc:
            raise GatewayError(
                "Document conversion exceeded the configured timeout.",
                status_code=408,
                code="document_timeout",
            ) from exc

    async def validate_image_url(self, url: str) -> None:
        await validate_document_url(url, self.url_policy)
