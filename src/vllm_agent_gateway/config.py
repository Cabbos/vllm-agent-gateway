from __future__ import annotations

import os
from dataclasses import dataclass


def _integer(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _floating(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(
        value.strip() for value in os.environ.get(name, default).split(",") if value.strip()
    )


@dataclass(frozen=True, slots=True)
class Settings:
    upstream: str
    served_model: str
    model_context_length: int
    model_family: str
    model_parameter_size: str
    model_quantization: str
    model_format: str
    model_size_bytes: int
    model_vram_bytes: int
    host: str
    port: int
    log_level: str
    api_keys: tuple[str, ...]
    cors_origins: tuple[str, ...]
    trusted_hosts: tuple[str, ...]
    max_request_bytes: int
    max_pdf_bytes: int
    max_pdf_pages: int
    max_rendered_pages: int
    max_extracted_chars: int
    upstream_connect_timeout: float

    @classmethod
    def from_env(cls) -> Settings:
        served_model = os.environ.get(
            "SERVED_MODEL",
            os.environ.get("LOCAL_SERVED_MODEL", "local-model"),
        )
        context_length = _integer(
            "MODEL_CONTEXT_LENGTH",
            _integer("LOCAL_MODEL_CONTEXT_LENGTH", 32768),
        )
        return cls(
            upstream=os.environ.get("VLLM_UPSTREAM", "http://127.0.0.1:8001").rstrip("/"),
            served_model=served_model,
            model_context_length=context_length,
            model_family=os.environ.get("MODEL_FAMILY", "local"),
            model_parameter_size=os.environ.get("MODEL_PARAMETER_SIZE", "unknown"),
            model_quantization=os.environ.get("MODEL_QUANTIZATION", "unknown"),
            model_format=os.environ.get("MODEL_FORMAT", "safetensors"),
            model_size_bytes=_integer("MODEL_SIZE_BYTES", 0),
            model_vram_bytes=_integer("MODEL_VRAM_BYTES", 0),
            host=os.environ.get("GATEWAY_HOST", "0.0.0.0"),
            port=_integer("GATEWAY_PORT", 8000),
            log_level=os.environ.get("GATEWAY_LOG_LEVEL", "info"),
            api_keys=_csv("GATEWAY_API_KEYS"),
            cors_origins=_csv("GATEWAY_CORS_ORIGINS", "*"),
            trusted_hosts=_csv("GATEWAY_TRUSTED_HOSTS", "*"),
            max_request_bytes=_integer("GATEWAY_MAX_REQUEST_BYTES", 64 * 1024 * 1024),
            max_pdf_bytes=_integer("PDF_COMPAT_MAX_BYTES", 50 * 1024 * 1024),
            max_pdf_pages=_integer("PDF_COMPAT_MAX_PAGES", 64),
            max_rendered_pages=_integer("PDF_COMPAT_MAX_RENDERED_PAGES", 24),
            max_extracted_chars=_integer("PDF_COMPAT_MAX_CHARS", 500_000),
            upstream_connect_timeout=_floating("UPSTREAM_CONNECT_TIMEOUT", 10.0),
        )


settings = Settings.from_env()
