"""Webhook URL safety checks — SSRF mitigation."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# Env vars permitted to supply webhook Bearer tokens (blocks GEMINI_/ANTHROPIC_/etc.).
VALID_WEBHOOK_AUTH_ENV: frozenset[str] = frozenset({"ESCRIBA_WEBHOOK_TOKEN"})

_PRIVATE_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


class WebhookUrlError(ValueError):
    """Raised when a webhook URL fails SSRF / scheme validation."""


def validate_webhook_auth_env(name: str) -> str:
    """Return ``name`` if it is an allowed webhook auth env var."""
    cleaned = name.strip()
    if cleaned not in VALID_WEBHOOK_AUTH_ENV:
        allowed = ", ".join(sorted(VALID_WEBHOOK_AUTH_ENV))
        raise WebhookUrlError(
            f"webhook auth_env must be one of: {allowed} (got {name!r})"
        )
    return cleaned


def _is_blocked_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return any(addr in net for net in _PRIVATE_NETWORKS)


def validate_webhook_url(url: str, *, allow_localhost: bool = False) -> str:
    """Validate webhook URL: HTTPS only; block private/loopback unless testing.

    Args:
        url: Configured webhook URL.
        allow_localhost: When True, permit loopback hosts (unit tests only).

    Returns:
        Normalized URL string.

    Raises:
        WebhookUrlError: On disallowed scheme or resolved private target.
    """
    cleaned = url.strip()
    if not cleaned:
        raise WebhookUrlError("webhook url must not be empty")

    parsed = urlparse(cleaned)
    if parsed.scheme.lower() != "https":
        raise WebhookUrlError("webhook url must use https")
    if not parsed.hostname:
        raise WebhookUrlError("webhook url must include a hostname")

    host = parsed.hostname.lower()
    if host == "localhost":
        if not allow_localhost:
            raise WebhookUrlError("webhook url must not target localhost")
        return cleaned

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise WebhookUrlError(f"webhook url hostname could not be resolved: {host}") from exc
        for info in infos:
            ip_str = info[4][0]
            try:
                resolved = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if _is_blocked_ip(resolved):
                raise WebhookUrlError(
                    f"webhook url resolves to a private or loopback address: {ip_str}"
                )
        return cleaned

    if _is_blocked_ip(addr) and not (allow_localhost and addr.is_loopback):
        raise WebhookUrlError("webhook url must not target a private or loopback address")
    return cleaned
