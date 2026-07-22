import json

import anyio
import httpx
import pytest
from starlette.requests import ClientDisconnect

from vllm_agent_gateway.adapters.ollama import stream_response as ollama_stream_response
from vllm_agent_gateway.application import _close_after_gemini_stream
from vllm_agent_gateway.proxy.streaming import UpstreamStreamingResponse
from vllm_agent_gateway.proxy.upstream import iter_and_close


class TrackingStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class CheckpointCloseStream(TrackingStream):
    def __init__(self):
        super().__init__([])
        self.close_started = False

    async def aclose(self):
        self.close_started = True
        await anyio.sleep(0)
        self.closed = True


def _scope(spec_version: str):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }


async def test_generic_proxy_closes_upstream_when_consumer_stops():
    stream = TrackingStream([b"first", b"second"])
    response = httpx.Response(200, stream=stream)
    iterator = iter_and_close(response)

    assert await anext(iterator) == b"first"
    await iterator.aclose()

    assert stream.closed is True


async def test_gemini_adapter_closes_upstream_when_consumer_stops():
    event = {
        "id": "chatcmpl-cancel",
        "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}],
    }
    stream = TrackingStream([f"data: {json.dumps(event)}\n\n".encode()])
    response = httpx.Response(200, stream=stream)
    iterator = _close_after_gemini_stream(
        response,
        framing="sse",
        model_version="local-model",
    )

    assert (await anext(iterator)).startswith(b"data: ")
    await iterator.aclose()

    assert stream.closed is True


async def test_ollama_adapter_closes_upstream_when_consumer_stops():
    event = {
        "choices": [{"delta": {"content": "partial"}, "finish_reason": None}],
    }
    stream = TrackingStream([f"data: {json.dumps(event)}\n\n".encode()])
    response = httpx.Response(200, stream=stream)
    iterator = ollama_stream_response(response, "local-model", "chat", 0)

    assert "partial" in await anext(iterator)
    await iterator.aclose()

    assert stream.closed is True


async def test_asgi_23_disconnect_finishes_checkpointing_upstream_close():
    stream = CheckpointCloseStream()
    upstream = httpx.Response(200, stream=stream)

    async def body():
        while True:
            yield b"chunk"
            await anyio.sleep(0)

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        await anyio.sleep(0)

    response = UpstreamStreamingResponse(body(), upstream_response=upstream)
    await response(_scope("2.3"), receive, send)

    assert stream.close_started is True
    assert stream.closed is True


async def test_asgi_24_send_error_finishes_checkpointing_upstream_close():
    stream = CheckpointCloseStream()
    upstream = httpx.Response(200, stream=stream)

    async def body():
        yield b"chunk"

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    response = UpstreamStreamingResponse(body(), upstream_response=upstream)
    with pytest.raises(ClientDisconnect):
        await response(_scope("2.4"), receive, send)

    assert stream.close_started is True
    assert stream.closed is True
