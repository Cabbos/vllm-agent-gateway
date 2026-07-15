from dataclasses import replace

from fastapi.testclient import TestClient

import vllm_agent_gateway.app as gateway


def test_root_reports_gateway_metadata():
    with TestClient(gateway.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "vLLM Agent Gateway"


def test_optional_api_key_rejects_unauthenticated_requests():
    configured = replace(gateway.settings, api_keys=("secret",))

    with TestClient(gateway.create_app(configured)) as client:
        response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["x-request-id"]


def test_health_endpoint_stays_public_when_auth_is_enabled():
    configured = replace(gateway.settings, api_keys=("secret",))

    with TestClient(gateway.create_app(configured)) as client:
        response = client.get("/")

    assert response.status_code == 200
