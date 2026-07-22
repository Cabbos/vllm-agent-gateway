from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from typing import Any

from .exceptions import (
    DocumentFormatError,
    DocumentSourceError,
    DocumentTooLargeError,
)
from .models import DocumentLimits, DocumentStats, LoadedDocument
from .url_security import (
    RemoteFetcher,
    RemoteFetchResult,
    RemoteURLPolicy,
    fetch_remote_document,
)

SUPPORTED_MEDIA_TYPES = {"application/pdf", "text/plain"}


def safe_document_title(block: Mapping[str, Any]) -> str:
    title = block.get("title") or block.get("name") or "attached document"
    cleaned = re.sub(r"[\r\n\t]+", " ", str(title)).strip()
    return cleaned[:200] or "attached document"


def check_raw_size(data: bytes, limits: DocumentLimits, *, label: str = "Document") -> bytes:
    if len(data) > limits.max_raw_bytes:
        raise DocumentTooLargeError(
            f"{label} exceeds the configured raw-size limit.",
            details={"actual": len(data), "limit": limits.max_raw_bytes},
        )
    return data


def decode_base64_data(
    encoded: Any,
    limits: DocumentLimits,
    *,
    label: str = "Document",
) -> bytes:
    if not isinstance(encoded, str):
        raise DocumentSourceError(f"{label} source.data must be a base64 string.")

    # Reject clearly oversized input before allocating the decoded buffer. A
    # four-character base64 group encodes no more than three raw bytes.
    max_encoded = ((limits.max_raw_bytes + 2) // 3) * 4
    if len(encoded) > max_encoded:
        raise DocumentTooLargeError(
            f"{label} exceeds the configured raw-size limit.",
            details={"limit": limits.max_raw_bytes},
        )
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DocumentSourceError(f"{label} source.data is not valid base64.") from exc
    return check_raw_size(decoded, limits, label=label)


async def load_document_source(
    block: Mapping[str, Any],
    *,
    limits: DocumentLimits | None = None,
    url_policy: RemoteURLPolicy | None = None,
    remote_fetcher: RemoteFetcher | None = None,
) -> LoadedDocument:
    """Load a supported document block into a bounded in-memory representation."""

    limits = limits or DocumentLimits()
    source = block.get("source")
    if not isinstance(source, Mapping):
        raise DocumentSourceError("Document block is missing a valid source object.")

    source_type = source.get("type")
    media_type = str(source.get("media_type") or "").lower()
    if source_type == "url" and not media_type:
        media_type = "application/pdf"
    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise DocumentSourceError("Documents must use application/pdf or text/plain media types.")

    title = safe_document_title(block)
    if source_type == "text":
        if media_type != "text/plain":
            raise DocumentSourceError("Text document sources must use text/plain.")
        value = source.get("data")
        if not isinstance(value, str):
            raise DocumentSourceError("Plain-text document source.data must be a string.")
        data = check_raw_size(value.encode("utf-8"), limits, label="Plain-text document")
        return LoadedDocument(data, media_type, "text", title)

    if source_type == "base64":
        data = decode_base64_data(source.get("data"), limits, label="Document")
        kind = "pdf_base64" if media_type == "application/pdf" else "text_base64"
        return LoadedDocument(data, media_type, kind, title)

    if source_type == "url":
        url = source.get("url")
        if not isinstance(url, str):
            raise DocumentSourceError("Document URL source.url must be a string.")
        if remote_fetcher is None:
            result = await fetch_remote_document(
                url,
                policy=url_policy or RemoteURLPolicy(),
                max_bytes=limits.max_raw_bytes,
            )
        else:
            result = await remote_fetcher(url, limits.max_raw_bytes)
        data = result.data if isinstance(result, RemoteFetchResult) else result
        if not isinstance(data, bytes):
            raise DocumentSourceError("Remote document fetcher must return bytes.")
        check_raw_size(data, limits, label="Remote document")
        kind = "pdf_url" if media_type == "application/pdf" else "text_url"
        return LoadedDocument(data, media_type, kind, title)

    if source_type == "file":
        raise DocumentSourceError(
            "File ID document sources require an external Files API. Send base64 or an "
            "explicitly allowed URL instead."
        )
    raise DocumentSourceError("Document source must use text, base64, or URL data.")


def convert_plain_document(
    document: LoadedDocument,
    *,
    limits: DocumentLimits | None = None,
) -> tuple[list[dict[str, Any]], DocumentStats]:
    limits = limits or DocumentLimits()
    if document.media_type != "text/plain":
        raise DocumentFormatError("Loaded document is not plain text.")
    try:
        text = document.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentFormatError("Plain-text documents must use UTF-8 encoding.") from exc

    truncated = len(text) > limits.max_extracted_chars
    text = text[: limits.max_extracted_chars]
    if truncated:
        text += "\n[Document text truncated.]"
    stats = DocumentStats(
        code="document_text",
        source_kind=document.source_kind,
        raw_bytes=document.raw_bytes,
        text_chars=min(len(text), limits.max_extracted_chars),
        truncated_pages=int(truncated),
    )
    return [{"type": "text", "text": f"[Document: {document.title}]\n{text}"}], stats
