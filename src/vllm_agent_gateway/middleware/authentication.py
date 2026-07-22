from __future__ import annotations

import hashlib
import secrets
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl

from starlette.types import ASGIApp, Receive, Scope, Send

from ._responses import send_json_error


@dataclass(frozen=True, slots=True)
class APIKeyCredential:
    value: str
    source: str


def _headers(scope: Scope) -> dict[bytes, bytes]:
    return {name.lower(): value for name, value in scope.get("headers", [])}


def extract_api_key(scope: Scope) -> APIKeyCredential | None:
    """Extract an API key using a deterministic, fail-closed precedence order."""
    headers = _headers(scope)
    authorization = headers.get(b"authorization")
    if authorization is not None:
        value = authorization.decode("latin-1").strip()
        scheme, separator, credentials = value.partition(" ")
        if separator and scheme.lower() == "bearer":
            return APIKeyCredential(credentials.strip(), "bearer")

    for header, source in (
        (b"x-api-key", "x-api-key"),
        (b"api-key", "api-key"),
        (b"x-goog-api-key", "x-goog-api-key"),
    ):
        if header in headers:
            return APIKeyCredential(headers[header].decode("latin-1").strip(), source)

    query_string = scope.get("query_string", b"")
    for name, value in parse_qsl(
        query_string.decode("latin-1"), keep_blank_values=True, strict_parsing=False
    ):
        if name == "key":
            return APIKeyCredential(value, "query")
    return None


def extract_api_key_value(scope: Scope) -> str:
    credential = extract_api_key(scope)
    return credential.value if credential is not None else ""


def api_key_is_valid(provided: str, expected_keys: Sequence[str]) -> bool:
    """Compare against every configured key without leaking the matching key index."""
    matched = 0
    for expected in expected_keys:
        matched |= int(secrets.compare_digest(provided, expected))
    return bool(provided) and bool(matched)


def api_key_fingerprint(api_key: str) -> str:
    """Return an opaque identifier suitable for internal rate-limit bucket lookup."""
    digest = hashlib.sha256(b"vllm-agent-gateway\0" + api_key.encode("utf-8")).hexdigest()
    return digest[:32]


class APIKeyAuthMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        api_keys: Sequence[str],
        public_paths: Collection[str] = ("/", "/healthz", "/readyz", "/v1/health"),
        allow_options: bool = True,
    ) -> None:
        self.app = app
        self.api_keys = tuple(api_keys)
        self.public_paths = frozenset(public_paths)
        self.allow_options = allow_options

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.api_keys:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        is_public = path in self.public_paths or (self.allow_options and method == "OPTIONS")
        if is_public:
            await self.app(scope, receive, send)
            return

        credential = extract_api_key(scope)
        provided = credential.value if credential is not None else ""
        if not api_key_is_valid(provided, self.api_keys):
            await send_json_error(
                send,
                status_code=401,
                message="Missing or invalid gateway API key.",
                headers=((b"www-authenticate", b"Bearer"),),
            )
            return

        state = scope.setdefault("state", {})
        state["gateway_api_key_id"] = api_key_fingerprint(provided)
        state["gateway_api_key_source"] = credential.source if credential is not None else ""
        await self.app(scope, receive, send)
