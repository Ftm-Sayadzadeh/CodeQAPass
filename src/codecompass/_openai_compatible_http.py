"""Small, secret-safe HTTP helper for OpenAI-compatible providers."""

from __future__ import annotations

import json
import http.client
import socket
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit


class OpenAICompatibleHTTPError(Exception):
    """An HTTP transport or response-decoding failure without request secrets."""

    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.message = message
        super().__init__(message)


def validate_base_url(base_url: str) -> str:
    """Return a normalized HTTP(S) base URL without embedded credentials or query data."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("OpenAI-compatible base_url is required")
    normalized = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OpenAI-compatible base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("OpenAI-compatible base_url must not contain credentials, query, or fragment")
    return normalized


def post_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    """POST JSON and return an object while keeping headers and response bodies out of errors."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url}/{endpoint.lstrip('/')}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise OpenAICompatibleHTTPError(
            "HTTPError", f"OpenAI-compatible request failed with HTTP {error.code}"
        ) from None
    except (socket.timeout, TimeoutError):
        raise OpenAICompatibleHTTPError("TimeoutError", "OpenAI-compatible request timed out") from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise OpenAICompatibleHTTPError("TimeoutError", "OpenAI-compatible request timed out") from None
        raise OpenAICompatibleHTTPError("ConnectionError", "OpenAI-compatible request failed") from None
    except (OSError, http.client.IncompleteRead):
        raise OpenAICompatibleHTTPError("ConnectionError", "OpenAI-compatible request failed") from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise OpenAICompatibleHTTPError(
            "invalid_response_encoding", "OpenAI-compatible response was not valid UTF-8"
        ) from None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        raise OpenAICompatibleHTTPError(
            "invalid_response_json", "OpenAI-compatible response was not valid JSON"
        ) from None
    if not isinstance(decoded, dict):
        raise OpenAICompatibleHTTPError(
            "invalid_response_top_level", "OpenAI-compatible response must be a JSON object"
        )
    return decoded
