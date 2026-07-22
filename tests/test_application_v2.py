import json
from dataclasses import replace

import httpx
from fastapi.testclient import TestClient

from vllm_agent_gateway.application import create_app
from vllm_agent_gateway.config import settings


def test_client_credentials_are_not_forwarded_to_vllm():
    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        seen["authorization"] = request.headers.get("authorization")
        seen["x-goog-api-key"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json={"object": "list", "data": []})

    config = replace(
        settings,
        upstream="http://upstream.test",
        upstream_api_key="backend-secret",
        api_keys=("client-secret",),
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        response = client.get("/v1/models?key=client-secret&api-version=2025-01-01")

    assert response.status_code == 200
    assert seen == {
        "query": {},
        "authorization": "Bearer backend-secret",
        "x-goog-api-key": None,
    }


def test_remote_documents_are_denied_before_upstream_by_default():
    upstream_calls = 0

    def upstream(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        return httpx.Response(200, json={})

    config = replace(
        settings,
        upstream="http://upstream.test",
        document_url_policy="deny",
        api_keys=(),
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "alias",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": "remote.pdf",
                                "file_url": "https://example.com/remote.pdf",
                            }
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert "disabled" in response.json()["error"]["message"]
    assert upstream_calls == 0


def test_gemini_streaming_is_converted_incrementally_to_sse():
    seen_payload = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal seen_payload
        seen_payload = json.loads(request.content)
        first = {
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "delta": {"content": "你"}, "finish_reason": None}],
        }
        second = {
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "delta": {"content": "好"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }
        content = (
            f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
            f"data: {json.dumps(second, ensure_ascii=False)}\n\n"
            "data: [DONE]\n\n"
        ).encode()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    config = replace(settings, upstream="http://upstream.test", api_keys=())
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        response = client.post(
            "/v1beta/models/alias:streamGenerateContent?alt=sse&key=not-forwarded",
            json={"contents": [{"role": "user", "parts": [{"text": "你好"}]}]},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert seen_payload["stream"] is True
    assert '"text":"你"' in response.text
    assert '"text":"好"' in response.text
    assert "usageMetadata" in response.text


def test_gateway_metrics_use_bounded_protocol_and_outcome_labels():
    def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    config = replace(settings, upstream="http://upstream.test", api_keys=(), metrics_enabled=True)
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 200
        metrics = client.get("/gateway/metrics")

    assert metrics.status_code == 200
    assert 'gateway_requests_total{protocol="openai",outcome="success"} 1' in metrics.text
    assert "request_id=" not in metrics.text


def test_disabled_gateway_metrics_are_not_forwarded_upstream():
    upstream_calls = 0

    def upstream(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        return httpx.Response(200, text="upstream")

    config = replace(settings, metrics_enabled=False, api_keys=())
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        response = client.get("/gateway/metrics")

    assert response.status_code == 404
    assert upstream_calls == 0


def test_upstream_network_details_are_not_returned_to_clients():
    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret internal endpoint 10.0.0.4:8123", request=request)

    config = replace(settings, upstream="http://upstream.test", api_keys=())
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 503
    assert "10.0.0.4" not in response.text
    assert "secret internal" not in response.text
