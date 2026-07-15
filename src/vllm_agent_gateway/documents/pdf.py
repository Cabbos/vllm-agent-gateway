from __future__ import annotations

import base64
import math
import re
from collections.abc import Mapping
from typing import Any

import fitz

from .exceptions import DocumentFormatError, DocumentLimitError, DocumentSourceError
from .loader import convert_plain_document, load_document_source
from .models import DocumentLimits, DocumentStats, LoadedDocument
from .url_security import RemoteFetcher, RemoteURLPolicy


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(data: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def _page_pixel_count(page: fitz.Page, scale: float) -> int:
    width = float(page.rect.width) * scale
    height = float(page.rect.height) * scale
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise DocumentFormatError("PDF page has invalid dimensions.")
    return math.ceil(width) * math.ceil(height)


def convert_loaded_pdf(
    loaded: LoadedDocument,
    *,
    limits: DocumentLimits | None = None,
) -> tuple[list[dict[str, Any]], DocumentStats]:
    """Convert searchable pages to text and sparse/scanned pages to JPEG blocks."""

    limits = limits or DocumentLimits()
    if loaded.media_type != "application/pdf":
        raise DocumentSourceError("Loaded document is not a PDF.")
    try:
        document = fitz.open(stream=loaded.data, filetype="pdf")
    except Exception as exc:
        raise DocumentFormatError("Unable to open PDF document.") from exc

    try:
        if document.needs_pass:
            raise DocumentFormatError("Password-protected PDFs are not supported.")
        if document.page_count == 0:
            raise DocumentFormatError("PDF contains no pages.")
        if document.page_count > limits.max_pdf_pages:
            raise DocumentLimitError(
                "PDF exceeds the configured page limit.",
                details={"actual": document.page_count, "limit": limits.max_pdf_pages},
            )

        output: list[dict[str, Any]] = []
        stats = DocumentStats(
            code=loaded.source_kind,
            source_kind=loaded.source_kind,
            raw_bytes=loaded.raw_bytes,
            pages=document.page_count,
        )
        for page_index in range(document.page_count):
            try:
                page = document.load_page(page_index)
                page_pixels = _page_pixel_count(page, limits.render_scale)
                if page_pixels > limits.max_page_pixels:
                    raise DocumentLimitError(
                        "PDF page exceeds the configured rendered-pixel limit.",
                        details={
                            "page": page_index + 1,
                            "actual": page_pixels,
                            "limit": limits.max_page_pixels,
                        },
                    )
                text = page.get_text("text", sort=True).replace("\x00", "").strip()
            except (DocumentFormatError, DocumentLimitError):
                raise
            except Exception as exc:
                raise DocumentFormatError(f"Unable to inspect PDF page {page_index + 1}.") from exc

            meaningful_chars = len(re.sub(r"\s+", "", text))
            page_label = f"[PDF: {loaded.title} - page {page_index + 1}/{document.page_count}]"
            if meaningful_chars >= limits.text_page_threshold:
                stats.searchable_pages += 1
                remaining = limits.max_extracted_chars - stats.text_chars
                if remaining <= 0:
                    stats.truncated_pages += 1
                    output.append(
                        _text_block(
                            f"{page_label}\n[Text omitted because the PDF extraction limit "
                            "was reached.]"
                        )
                    )
                    continue

                page_text = text[:remaining]
                stats.text_chars += len(page_text)
                if len(page_text) < len(text):
                    stats.truncated_pages += 1
                    page_text += "\n[Page text truncated.]"
                output.append(_text_block(f"{page_label}\n{page_text}"))
                continue

            if stats.rendered_pages >= limits.max_rendered_pages:
                raise DocumentLimitError(
                    "PDF exceeds the configured scanned-page rendering limit.",
                    details={"limit": limits.max_rendered_pages},
                )

            output.append(_text_block(f"{page_label}\n[Scanned page rendered as an image.]"))
            try:
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(limits.render_scale, limits.render_scale),
                    alpha=False,
                )
                if pixmap.width * pixmap.height > limits.max_page_pixels:
                    raise DocumentLimitError(
                        "Rendered PDF page exceeds the configured pixel limit.",
                        details={
                            "page": page_index + 1,
                            "actual": pixmap.width * pixmap.height,
                            "limit": limits.max_page_pixels,
                        },
                    )
                image = pixmap.tobytes("jpeg", jpg_quality=limits.jpeg_quality)
            except DocumentLimitError:
                raise
            except Exception as exc:
                raise DocumentFormatError(f"Unable to render PDF page {page_index + 1}.") from exc
            output.append(_image_block(image))
            stats.rendered_pages += 1
        return output, stats
    finally:
        document.close()


async def convert_pdf_document(
    block: Mapping[str, Any],
    *,
    limits: DocumentLimits | None = None,
    url_policy: RemoteURLPolicy | None = None,
    remote_fetcher: RemoteFetcher | None = None,
) -> tuple[list[dict[str, Any]], DocumentStats]:
    limits = limits or DocumentLimits()
    loaded = await load_document_source(
        block,
        limits=limits,
        url_policy=url_policy,
        remote_fetcher=remote_fetcher,
    )
    return convert_loaded_pdf(loaded, limits=limits)


async def convert_document(
    block: Mapping[str, Any],
    *,
    limits: DocumentLimits | None = None,
    url_policy: RemoteURLPolicy | None = None,
    remote_fetcher: RemoteFetcher | None = None,
) -> tuple[list[dict[str, Any]], DocumentStats]:
    limits = limits or DocumentLimits()
    loaded = await load_document_source(
        block,
        limits=limits,
        url_policy=url_policy,
        remote_fetcher=remote_fetcher,
    )
    if loaded.media_type == "application/pdf":
        return convert_loaded_pdf(loaded, limits=limits)
    return convert_plain_document(loaded, limits=limits)
