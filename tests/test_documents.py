import base64

import fitz
import pytest

from vllm_agent_gateway.app import (
    AnthropicCompatibilityError,
    _convert_openai_documents,
    _validate_public_document_url,
    convert_pdf_document,
)


def _pdf_base64(text: str | None = None) -> str:
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_textbox(fitz.Rect(72, 72, 500, 200), text, fontsize=12)
    raw = document.tobytes()
    document.close()
    return base64.b64encode(raw).decode("ascii")


def test_searchable_pdf_becomes_responses_text():
    marker = "A searchable PDF sentence that is deliberately longer than forty characters."
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Read the file"},
                    {
                        "type": "input_file",
                        "filename": "sample.pdf",
                        "file_data": f"data:application/pdf;base64,{_pdf_base64(marker)}",
                    },
                ],
            }
        ]
    }
    events = []

    _convert_openai_documents(payload, events)

    parts = payload["input"][0]["content"]
    assert [part["type"] for part in parts] == ["input_text", "input_text"]
    assert marker in parts[1]["text"]
    assert events[0]["code"] == "openai_input_file_pdf"


def test_scanned_pdf_becomes_an_image():
    blocks, stats = convert_pdf_document(
        {
            "type": "document",
            "title": "scan.pdf",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": _pdf_base64(),
            },
        }
    )

    assert [block["type"] for block in blocks] == ["text", "image"]
    assert stats["rendered_pages"] == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/file.pdf",
        "http://localhost/file.pdf",
        "http://metadata.internal/file.pdf",
        "file:///tmp/file.pdf",
    ],
)
def test_document_url_blocks_internal_targets(url):
    with pytest.raises(AnthropicCompatibilityError):
        _validate_public_document_url(url)
