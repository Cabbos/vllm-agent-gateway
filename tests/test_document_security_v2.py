import base64

import fitz
import httpx
import pytest

from vllm_agent_gateway.documents import (
    DocumentLimitError,
    DocumentLimits,
    DocumentTooLargeError,
    RemoteDocumentDeniedError,
    RemoteURLPolicy,
    convert_document,
    decode_base64_data,
    fetch_remote_document,
    validate_document_url,
)


def _pdf_bytes(*page_texts: str | None, page_size: tuple[int, int] | None = None) -> bytes:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page(
            width=page_size[0] if page_size else 595,
            height=page_size[1] if page_size else 842,
        )
        if text:
            page.insert_textbox(fitz.Rect(72, 72, 520, 300), text, fontsize=12)
    raw = document.tobytes()
    document.close()
    return raw


def _pdf_block(raw: bytes, title: str = "sample.pdf") -> dict:
    return {
        "type": "document",
        "title": title,
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(raw).decode("ascii"),
        },
    }


@pytest.mark.asyncio
async def test_remote_urls_are_denied_by_default():
    with pytest.raises(RemoteDocumentDeniedError, match="disabled"):
        await validate_document_url(
            "https://documents.example/file.pdf",
            RemoteURLPolicy(),
            resolver=lambda _host, _port: ("93.184.216.34",),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "169.254.169.254", "198.18.0.1"])
async def test_private_internal_and_benchmark_addresses_are_not_implicitly_allowed(address):
    policy = RemoteURLPolicy(mode="allowlist", allowed_hosts=("documents.example",))

    with pytest.raises(RemoteDocumentDeniedError, match="outside the allowed networks"):
        await validate_document_url(
            "https://documents.example/file.pdf",
            policy,
            resolver=lambda _host, _port: (address,),
        )


@pytest.mark.asyncio
async def test_private_network_requires_explicit_network_opt_in():
    policy = RemoteURLPolicy(
        mode="allowlist",
        allowed_hosts=("documents.example",),
        extra_allowed_networks=("10.20.0.0/16",),
    )

    result = await validate_document_url(
        "https://documents.example/file.pdf",
        policy,
        resolver=lambda _host, _port: ("10.20.4.8",),
    )

    assert str(result.addresses[0]) == "10.20.4.8"


@pytest.mark.asyncio
async def test_redirect_target_is_resolved_and_revalidated():
    requested_hosts = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        return httpx.Response(302, headers={"location": "http://internal.example/secret.pdf"})

    def resolver(host: str, _port: int):
        return ("93.184.216.34",) if host == "public.example" else ("10.0.0.7",)

    policy = RemoteURLPolicy(
        mode="allowlist",
        allowed_hosts=("public.example", "internal.example"),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RemoteDocumentDeniedError, match="outside the allowed networks"):
            await fetch_remote_document(
                "https://public.example/file.pdf",
                policy=policy,
                max_bytes=1024,
                resolver=resolver,
                client=client,
            )

    assert requested_hosts == ["public.example"]


@pytest.mark.asyncio
async def test_actual_peer_ip_must_match_validated_dns_result():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"pdf", extensions={"peer_ip": "127.0.0.1"})

    policy = RemoteURLPolicy(mode="allowlist", allowed_hosts=("documents.example",))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RemoteDocumentDeniedError, match="peer address"):
            await fetch_remote_document(
                "https://documents.example/file.pdf",
                policy=policy,
                max_bytes=1024,
                resolver=lambda _host, _port: ("93.184.216.34",),
                client=client,
            )


def test_base64_is_rejected_before_oversized_data_is_accepted():
    limits = DocumentLimits(max_raw_bytes=3)

    with pytest.raises(DocumentTooLargeError) as exc_info:
        decode_base64_data(base64.b64encode(b"four").decode("ascii"), limits)

    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_pdf_page_count_limit_is_enforced():
    with pytest.raises(DocumentLimitError) as exc_info:
        await convert_document(
            _pdf_block(_pdf_bytes("first page", "second page")),
            limits=DocumentLimits(max_pdf_pages=1),
        )

    assert exc_info.value.details == {"actual": 2, "limit": 1}


@pytest.mark.asyncio
async def test_pdf_page_pixel_limit_is_checked_before_rendering():
    with pytest.raises(DocumentLimitError, match="rendered-pixel"):
        await convert_document(
            _pdf_block(_pdf_bytes(None, page_size=(1000, 1000))),
            limits=DocumentLimits(max_page_pixels=1_000_000),
        )


@pytest.mark.asyncio
async def test_scanned_page_rendering_can_be_disabled():
    with pytest.raises(DocumentLimitError, match="scanned-page"):
        await convert_document(
            _pdf_block(_pdf_bytes(None)),
            limits=DocumentLimits(max_rendered_pages=0),
        )


@pytest.mark.asyncio
async def test_searchable_and_scanned_pdf_pages_use_bounded_conversions():
    marker = "A searchable PDF sentence deliberately longer than the text threshold."
    blocks, stats = await convert_document(_pdf_block(_pdf_bytes(marker, None)))

    assert [block["type"] for block in blocks] == ["text", "text", "image"]
    assert marker in blocks[0]["text"]
    assert stats.pages == 2
    assert stats.searchable_pages == 1
    assert stats.rendered_pages == 1
    assert stats.raw_bytes > 0


@pytest.mark.asyncio
async def test_plain_text_conversion_reports_character_truncation():
    block = {
        "type": "document",
        "title": "notes.txt",
        "source": {"type": "text", "media_type": "text/plain", "data": "abcdef"},
    }
    blocks, stats = await convert_document(
        block,
        limits=DocumentLimits(max_extracted_chars=4),
    )

    assert "abcd\n[Document text truncated.]" in blocks[0]["text"]
    assert stats.text_chars == 4
    assert stats.truncated_pages == 1
