"""Webhook knowledge-store adapter — POST session export as JSON."""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from escriba.knowledge.constants import EXPORT_TIMEOUT_CAP_SECONDS
from escriba.knowledge.payload import build_export_payload
from escriba.knowledge.port import KnowledgeStore
from escriba.knowledge.url_safety import (
    validate_webhook_auth_env,
    validate_webhook_url,
)

logger = logging.getLogger(__name__)


class WebhookAdapter(KnowledgeStore):
    """POST session payload to a configured HTTPS endpoint."""

    def __init__(
        self,
        url: str,
        *,
        auth_env: str = "ESCRIBA_WEBHOOK_TOKEN",
        timeout_seconds: float = EXPORT_TIMEOUT_CAP_SECONDS,
        allow_localhost: bool = False,
    ) -> None:
        self._url = validate_webhook_url(url, allow_localhost=allow_localhost)
        self._auth_env = validate_webhook_auth_env(auth_env)
        self._timeout = min(timeout_seconds, EXPORT_TIMEOUT_CAP_SECONDS)

    def export(
        self,
        session: dict[str, Any],
        summary_json: dict[str, Any] | None,
        audio_path: Path | None,
        segments: list[dict[str, Any]] | None = None,
    ) -> None:
        payload = build_export_payload(session, summary_json, audio_path, segments)
        headers = {"Content-Type": "application/json"}
        token = os.getenv(self._auth_env, "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        body = json.dumps(payload, default=str).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                logger.info(
                    "KnowledgeStore webhook: exported session %s (HTTP %s)",
                    session.get("id"),
                    response.status,
                )
        except urllib.error.HTTPError as exc:
            logger.error(
                "KnowledgeStore webhook: HTTP %s for session %s",
                exc.code,
                session.get("id"),
                exc_info=True,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.error(
                "KnowledgeStore webhook: export failed for session %s: %s",
                session.get("id"),
                exc,
                exc_info=True,
            )
