from vllm_agent_gateway.config import Settings


def test_settings_support_generic_and_legacy_names(monkeypatch):
    monkeypatch.setenv("SERVED_MODEL", "test-model")
    monkeypatch.setenv("MODEL_CONTEXT_LENGTH", "65536")
    monkeypatch.setenv("GATEWAY_API_KEYS", "first, second")
    monkeypatch.setenv("GATEWAY_MAX_PROMPT_IMAGES", "6")

    settings = Settings.from_env()

    assert settings.served_model == "test-model"
    assert settings.model_context_length == 65536
    assert settings.api_keys == ("first", "second")
    assert settings.max_prompt_images == 6


def test_legacy_model_environment_names(monkeypatch):
    monkeypatch.delenv("SERVED_MODEL", raising=False)
    monkeypatch.delenv("MODEL_CONTEXT_LENGTH", raising=False)
    monkeypatch.setenv("LOCAL_SERVED_MODEL", "legacy-model")
    monkeypatch.setenv("LOCAL_MODEL_CONTEXT_LENGTH", "4096")

    settings = Settings.from_env()

    assert settings.served_model == "legacy-model"
    assert settings.model_context_length == 4096
