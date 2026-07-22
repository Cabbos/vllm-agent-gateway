from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit


def _integer(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _floating(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _boolean(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(
        value.strip() for value in os.environ.get(name, default).split(",") if value.strip()
    )


def _url_policy(name: str = "DOCUMENT_URL_POLICY") -> Literal["deny", "allowlist"]:
    value = os.environ.get(name, "deny").strip().lower()
    if value not in {"deny", "allowlist"}:
        raise ValueError(f"{name} must be 'deny' or 'allowlist'")
    return "allowlist" if value == "allowlist" else "deny"


@dataclass(frozen=True, slots=True)
class Settings:
    upstream: str
    upstream_api_key: str
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
    max_prompt_images: int
    max_pdf_bytes: int
    max_pdf_pages: int
    max_rendered_pages: int
    max_extracted_chars: int
    pdf_max_page_pixels: int
    pdf_conversion_concurrency: int
    pdf_conversion_timeout_seconds: float
    document_url_policy: Literal["deny", "allowlist"]
    document_allowed_hosts: tuple[str, ...]
    document_extra_allowed_networks: tuple[str, ...]
    max_inflight: int
    max_queue_size: int
    queue_timeout_seconds: float
    requests_per_minute: float
    rate_limit_burst: int
    metrics_enabled: bool
    upstream_connect_timeout: float
    upstream_max_connections: int
    upstream_max_keepalive_connections: int

    def __post_init__(self) -> None:
        upstream = urlsplit(self.upstream)
        if upstream.scheme not in {"http", "https"} or not upstream.hostname:
            raise ValueError("VLLM_UPSTREAM must be an http:// or https:// URL")
        if not self.served_model.strip():
            raise ValueError("SERVED_MODEL cannot be empty")
        if not self.host.strip():
            raise ValueError("GATEWAY_HOST cannot be empty")
        positive = {
            "model_context_length": self.model_context_length,
            "max_request_bytes": self.max_request_bytes,
            "max_pdf_bytes": self.max_pdf_bytes,
            "max_pdf_pages": self.max_pdf_pages,
            "pdf_max_page_pixels": self.pdf_max_page_pixels,
            "pdf_conversion_concurrency": self.pdf_conversion_concurrency,
            "pdf_conversion_timeout_seconds": self.pdf_conversion_timeout_seconds,
            "queue_timeout_seconds": self.queue_timeout_seconds,
            "rate_limit_burst": self.rate_limit_burst,
            "upstream_connect_timeout": self.upstream_connect_timeout,
            "upstream_max_connections": self.upstream_max_connections,
            "upstream_max_keepalive_connections": self.upstream_max_keepalive_connections,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Gateway settings must be positive: {', '.join(invalid)}")
        non_negative = {
            "max_prompt_images": self.max_prompt_images,
            "max_rendered_pages": self.max_rendered_pages,
            "max_extracted_chars": self.max_extracted_chars,
            "max_inflight": self.max_inflight,
            "max_queue_size": self.max_queue_size,
            "requests_per_minute": self.requests_per_minute,
        }
        invalid = [name for name, value in non_negative.items() if value < 0]
        if invalid:
            raise ValueError(f"Gateway settings cannot be negative: {', '.join(invalid)}")
        if not 1 <= self.port <= 65535:
            raise ValueError("GATEWAY_PORT must be between 1 and 65535")
        if self.document_url_policy not in {"deny", "allowlist"}:
            raise ValueError("DOCUMENT_URL_POLICY must be 'deny' or 'allowlist'")
        if self.upstream_max_keepalive_connections > self.upstream_max_connections:
            raise ValueError(
                "UPSTREAM_MAX_KEEPALIVE_CONNECTIONS cannot exceed UPSTREAM_MAX_CONNECTIONS"
            )

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
        max_pdf_bytes = _integer("PDF_COMPAT_MAX_BYTES", 50 * 1024 * 1024)
        # Base64 expands raw files by ~4/3. Leave additional room for JSON,
        # prompts, tools and images so the document and request limits agree.
        default_request_bytes = (max_pdf_bytes * 4 + 2) // 3 + 4 * 1024 * 1024
        return cls(
            upstream=os.environ.get("VLLM_UPSTREAM", "http://127.0.0.1:8001").rstrip("/"),
            upstream_api_key=os.environ.get("VLLM_UPSTREAM_API_KEY", "").strip(),
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
            max_request_bytes=_integer("GATEWAY_MAX_REQUEST_BYTES", default_request_bytes),
            max_prompt_images=_integer(
                "GATEWAY_MAX_PROMPT_IMAGES",
                _integer("LOCAL_MAX_PROMPT_IMAGES", 4),
            ),
            max_pdf_bytes=max_pdf_bytes,
            max_pdf_pages=_integer("PDF_COMPAT_MAX_PAGES", 64),
            max_rendered_pages=_integer("PDF_COMPAT_MAX_RENDERED_PAGES", 24),
            max_extracted_chars=_integer("PDF_COMPAT_MAX_CHARS", 500_000),
            pdf_max_page_pixels=_integer("PDF_MAX_PAGE_PIXELS", 16_000_000),
            pdf_conversion_concurrency=_integer("PDF_CONVERSION_CONCURRENCY", 2),
            pdf_conversion_timeout_seconds=_floating("PDF_CONVERSION_TIMEOUT_SECONDS", 60.0),
            document_url_policy=_url_policy(),
            document_allowed_hosts=_csv("DOCUMENT_ALLOWED_HOSTS"),
            document_extra_allowed_networks=_csv("DOCUMENT_EXTRA_ALLOWED_NETWORKS"),
            max_inflight=_integer("GATEWAY_MAX_INFLIGHT", 0),
            max_queue_size=_integer("GATEWAY_MAX_QUEUE_SIZE", 0),
            queue_timeout_seconds=_floating("GATEWAY_QUEUE_TIMEOUT_SECONDS", 30.0),
            requests_per_minute=_floating("GATEWAY_REQUESTS_PER_MINUTE", 0.0),
            rate_limit_burst=_integer("GATEWAY_RATE_LIMIT_BURST", 10),
            metrics_enabled=_boolean("GATEWAY_METRICS_ENABLED", True),
            upstream_connect_timeout=_floating("UPSTREAM_CONNECT_TIMEOUT", 10.0),
            upstream_max_connections=_integer("UPSTREAM_MAX_CONNECTIONS", 100),
            upstream_max_keepalive_connections=_integer("UPSTREAM_MAX_KEEPALIVE_CONNECTIONS", 20),
        )


settings = Settings.from_env()
