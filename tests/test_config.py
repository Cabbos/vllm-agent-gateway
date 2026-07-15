import pytest

from vllm_agent_gateway.config import Settings


def test_settings_support_generic_and_legacy_names(monkeypatch):
    monkeypatch.setenv("SERVED_MODEL", "test-model")
    monkeypatch.setenv("MODEL_CONTEXT_LENGTH", "65536")
    monkeypatch.setenv("GATEWAY_API_KEYS", "first, second")

    settings = Settings.from_env()

    assert settings.served_model == "test-model"
    assert settings.model_context_length == 65536
    assert settings.api_keys == ("first", "second")


def test_default_request_limit_can_hold_a_maximum_base64_pdf(monkeypatch):
    monkeypatch.setenv("PDF_COMPAT_MAX_BYTES", str(9 * 1024 * 1024))
    monkeypatch.delenv("GATEWAY_MAX_REQUEST_BYTES", raising=False)

    settings = Settings.from_env()

    assert settings.max_request_bytes > (settings.max_pdf_bytes * 4 + 2) // 3


def test_upstream_and_client_keys_are_configured_separately(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEYS", "client-one,client-two")
    monkeypatch.setenv("VLLM_UPSTREAM_API_KEY", "backend-only")

    settings = Settings.from_env()

    assert settings.api_keys == ("client-one", "client-two")
    assert settings.upstream_api_key == "backend-only"


def test_invalid_boolean_fails_fast(monkeypatch):
    monkeypatch.setenv("GATEWAY_METRICS_ENABLED", "sometimes")

    with pytest.raises(ValueError, match="boolean"):
        Settings.from_env()


def test_invalid_admission_limit_fails_fast(monkeypatch):
    monkeypatch.setenv("GATEWAY_MAX_INFLIGHT", "-1")

    with pytest.raises(ValueError, match="cannot be negative"):
        Settings.from_env()


def test_legacy_model_environment_names(monkeypatch):
    monkeypatch.delenv("SERVED_MODEL", raising=False)
    monkeypatch.delenv("MODEL_CONTEXT_LENGTH", raising=False)
    monkeypatch.setenv("LOCAL_SERVED_MODEL", "legacy-model")
    monkeypatch.setenv("LOCAL_MODEL_CONTEXT_LENGTH", "4096")

    settings = Settings.from_env()

    assert settings.served_model == "legacy-model"
    assert settings.model_context_length == 4096
