from __future__ import annotations

import asyncio
import inspect
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias
from urllib.parse import urljoin, urlsplit

import httpx

from .exceptions import (
    DocumentTooLargeError,
    RemoteDocumentDeniedError,
    RemoteDocumentFetchError,
)

IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork: TypeAlias = ipaddress.IPv4Network | ipaddress.IPv6Network
ResolverResult: TypeAlias = Iterable[str] | Awaitable[Iterable[str]]
Resolver: TypeAlias = Callable[[str, int], ResolverResult]


class RemoteFetcher(Protocol):
    async def __call__(self, url: str, max_bytes: int) -> bytes | RemoteFetchResult: ...


@dataclass(frozen=True, slots=True)
class RemoteURLPolicy:
    """Opt-in URL policy.

    ``deny`` rejects every remote source. ``allowlist`` requires the host to
    match ``allowed_hosts`` and every resolved/connected address to be public
    or explicitly covered by ``extra_allowed_networks``.
    """

    mode: Literal["deny", "allowlist"] = "deny"
    allowed_hosts: tuple[str, ...] = ()
    extra_allowed_networks: tuple[IPNetwork | str, ...] = ()
    allowed_ports: tuple[int, ...] = (80, 443)
    max_redirects: int = 5
    require_peer_ip: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"deny", "allowlist"}:
            raise ValueError("Remote URL policy must be 'deny' or 'allowlist'")
        if self.max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")
        if not self.allowed_ports or any(not 1 <= port <= 65535 for port in self.allowed_ports):
            raise ValueError("allowed_ports must contain valid TCP ports")

        hosts = tuple(_normalize_host_pattern(host) for host in self.allowed_hosts)
        networks = tuple(
            ipaddress.ip_network(value, strict=True) for value in self.extra_allowed_networks
        )
        object.__setattr__(self, "allowed_hosts", hosts)
        object.__setattr__(self, "extra_allowed_networks", networks)


@dataclass(frozen=True, slots=True)
class ValidatedURL:
    url: str
    hostname: str
    port: int
    addresses: tuple[IPAddress, ...]


@dataclass(frozen=True, slots=True)
class RemoteFetchResult:
    data: bytes
    final_url: str
    redirects: int
    peer_ip: str | None = None


def _normalize_hostname(hostname: str) -> str:
    cleaned = hostname.rstrip(".").lower()
    if not cleaned:
        raise ValueError("Host cannot be empty")
    try:
        return cleaned.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("Host is not valid IDNA") from exc


def _normalize_host_pattern(pattern: str) -> str:
    cleaned = str(pattern).strip()
    if cleaned.startswith("*."):
        return f"*.{_normalize_hostname(cleaned[2:])}"
    return _normalize_hostname(cleaned)


def _host_matches(hostname: str, pattern: str) -> bool:
    if pattern.startswith("*."):
        suffix = pattern[2:]
        return hostname != suffix and hostname.endswith(f".{suffix}")
    return hostname == pattern


def _ip_is_allowed(address: IPAddress, policy: RemoteURLPolicy) -> bool:
    if any(address in network for network in policy.extra_allowed_networks):
        return True
    return bool(
        address.is_global
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
    )


async def _default_resolver(hostname: str, port: int) -> Sequence[str]:
    try:
        items = await asyncio.get_running_loop().getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise RemoteDocumentDeniedError("Unable to resolve document URL host.") from exc
    return tuple(item[4][0] for item in items)


async def _resolve(resolver: Resolver, hostname: str, port: int) -> tuple[str, ...]:
    try:
        result = resolver(hostname, port)
        if inspect.isawaitable(result):
            result = await result
        return tuple(dict.fromkeys(result))
    except RemoteDocumentDeniedError:
        raise
    except (OSError, ValueError) as exc:
        raise RemoteDocumentDeniedError("Unable to resolve document URL host.") from exc


async def validate_document_url(
    url: str,
    policy: RemoteURLPolicy,
    *,
    resolver: Resolver | None = None,
) -> ValidatedURL:
    """Validate syntax, hostname policy, DNS results, and address classes."""

    if policy.mode == "deny":
        raise RemoteDocumentDeniedError("Remote document URLs are disabled.")
    if not isinstance(url, str):
        raise RemoteDocumentDeniedError("Document URL must be a string.")

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise RemoteDocumentDeniedError("Document URL must use http:// or https://.")
    if parsed.username is not None or parsed.password is not None:
        raise RemoteDocumentDeniedError("Credentials are not allowed in document URLs.")

    try:
        hostname = _normalize_hostname(parsed.hostname)
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except (UnicodeError, ValueError) as exc:
        raise RemoteDocumentDeniedError("Document URL contains an invalid host or port.") from exc

    if port not in policy.allowed_ports:
        raise RemoteDocumentDeniedError("Document URL uses a port that is not allowed.")
    if not any(_host_matches(hostname, pattern) for pattern in policy.allowed_hosts):
        raise RemoteDocumentDeniedError("Document URL host is not allowlisted.")

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        address_strings = await _resolve(resolver or _default_resolver, hostname, port)
    else:
        address_strings = (str(literal),)

    if not address_strings:
        raise RemoteDocumentDeniedError("Document URL host did not resolve to an address.")

    try:
        addresses = tuple(ipaddress.ip_address(value) for value in address_strings)
    except ValueError as exc:
        raise RemoteDocumentDeniedError(
            "Document URL host resolved to an invalid address."
        ) from exc
    if any(not _ip_is_allowed(address, policy) for address in addresses):
        raise RemoteDocumentDeniedError(
            "Document URL resolved to an address outside the allowed networks."
        )
    return ValidatedURL(url=url, hostname=hostname, port=port, addresses=addresses)


def _response_peer_ip(response: httpx.Response) -> IPAddress | None:
    direct = response.extensions.get("peer_ip")
    if direct is not None:
        try:
            return ipaddress.ip_address(str(direct))
        except ValueError:
            return None

    stream = response.extensions.get("network_stream")
    if stream is None or not hasattr(stream, "get_extra_info"):
        return None
    for key in ("server_addr", "peername", "remote_address"):
        try:
            value = stream.get_extra_info(key)
        except (OSError, RuntimeError):
            continue
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        if isinstance(value, str):
            try:
                return ipaddress.ip_address(value.split("%", 1)[0])
            except ValueError:
                continue
    return None


def _validate_response_peer(
    response: httpx.Response,
    validated: ValidatedURL,
    policy: RemoteURLPolicy,
) -> IPAddress | None:
    peer_ip = _response_peer_ip(response)
    if peer_ip is None:
        if policy.require_peer_ip:
            raise RemoteDocumentDeniedError(
                "The HTTP transport did not expose the document server peer address."
            )
        return None
    if not _ip_is_allowed(peer_ip, policy) or peer_ip not in validated.addresses:
        raise RemoteDocumentDeniedError(
            "Document server peer address did not match the validated DNS addresses."
        )
    return peer_ip


async def fetch_remote_document(
    url: str,
    *,
    policy: RemoteURLPolicy,
    max_bytes: int,
    resolver: Resolver | None = None,
    client: httpx.AsyncClient | None = None,
) -> RemoteFetchResult:
    """Fetch a bounded remote document, revalidating every redirect target."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=False,
            headers={"user-agent": "vllm-agent-gateway/0.2"},
        )

    current_url = url
    try:
        for redirect_count in range(policy.max_redirects + 1):
            validated = await validate_document_url(current_url, policy, resolver=resolver)
            try:
                async with client.stream("GET", current_url) as response:
                    peer_ip = _validate_response_peer(response, validated, policy)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise RemoteDocumentFetchError(
                                "Document URL redirect did not include a Location header."
                            )
                        current_url = urljoin(current_url, location)
                        continue

                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise RemoteDocumentFetchError(
                            f"Document URL returned HTTP {response.status_code}."
                        ) from exc

                    declared_size = response.headers.get("content-length")
                    if declared_size:
                        try:
                            parsed_size = int(declared_size)
                        except ValueError:
                            parsed_size = 0
                        if parsed_size > max_bytes:
                            raise DocumentTooLargeError(
                                "Remote document exceeds the configured raw-size limit.",
                                details={"limit": max_bytes},
                            )

                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > max_bytes:
                            raise DocumentTooLargeError(
                                "Remote document exceeds the configured raw-size limit.",
                                details={"limit": max_bytes},
                            )
                    return RemoteFetchResult(
                        data=bytes(data),
                        final_url=current_url,
                        redirects=redirect_count,
                        peer_ip=str(peer_ip) if peer_ip else None,
                    )
            except (DocumentTooLargeError, RemoteDocumentDeniedError, RemoteDocumentFetchError):
                raise
            except httpx.RequestError as exc:
                raise RemoteDocumentFetchError("Unable to fetch document URL.") from exc

        raise RemoteDocumentFetchError(
            f"Document URL exceeded the {policy.max_redirects}-redirect limit."
        )
    finally:
        if owns_client:
            await client.aclose()
