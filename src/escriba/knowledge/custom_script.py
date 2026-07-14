"""Custom-script knowledge-store adapter — invoke a user script with argv."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from escriba.knowledge.constants import EXPORT_TIMEOUT_CAP_SECONDS
from escriba.knowledge.payload import build_export_payload
from escriba.knowledge.port import KnowledgeStore

logger = logging.getLogger(__name__)


class CustomScriptPathError(ValueError):
    """Raised when a script path escapes the configured scripts directory."""


def resolve_script_path(script_path: str, scripts_dir: str) -> Path:
    """Resolve and jail ``script_path`` under ``scripts_dir``.

    Args:
        script_path: Relative or absolute script path from config.
        scripts_dir: Allowed root directory for export scripts.

    Returns:
        Resolved absolute path to the script.

    Raises:
        CustomScriptPathError: When the path escapes the jail or is missing.
    """
    candidate = _jailed_script_candidate(script_path, scripts_dir)
    if not candidate.is_file():
        raise CustomScriptPathError(f"custom-script not found: {candidate}")
    return candidate


def validate_script_path_config(script_path: str, scripts_dir: str) -> None:
    """Ensure a configured script path stays inside the scripts directory."""
    _jailed_script_candidate(script_path, scripts_dir)


def _jailed_script_candidate(script_path: str, scripts_dir: str) -> Path:
    root = Path(scripts_dir).expanduser().resolve()
    raw = Path(script_path).expanduser()
    candidate = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CustomScriptPathError(
            f"custom-script path must stay under {root} (got {candidate})"
        ) from exc
    return candidate


class CustomScriptAdapter(KnowledgeStore):
    """Run a configured executable with JSON on stdin (argv, not shell)."""

    def __init__(
        self,
        script_path: str,
        *,
        scripts_dir: str = "~/Library/Application Support/Escriba/scripts",
        timeout_seconds: float = EXPORT_TIMEOUT_CAP_SECONDS,
    ) -> None:
        self._script_path = script_path.strip()
        self._scripts_dir = scripts_dir
        self._timeout = min(timeout_seconds, EXPORT_TIMEOUT_CAP_SECONDS)

    def export(
        self,
        session: dict[str, Any],
        summary_json: dict[str, Any] | None,
        audio_path: Path | None,
        segments: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._script_path:
            logger.warning("KnowledgeStore custom-script: path is empty; skipping export")
            return
        try:
            script = resolve_script_path(self._script_path, self._scripts_dir)
        except CustomScriptPathError as exc:
            logger.error("KnowledgeStore custom-script: %s", exc)
            return
        payload = build_export_payload(session, summary_json, audio_path, segments)
        try:
            completed = subprocess.run(
                [str(script)],
                input=json.dumps(payload, default=str),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            if completed.returncode != 0:
                logger.error(
                    "KnowledgeStore custom-script: exit %s for session %s: %s",
                    completed.returncode,
                    session.get("id"),
                    (completed.stderr or completed.stdout or "").strip(),
                )
                return
            logger.info(
                "KnowledgeStore custom-script: exported session %s",
                session.get("id"),
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "KnowledgeStore custom-script: timed out after %ss for session %s",
                self._timeout,
                session.get("id"),
            )
        except OSError as exc:
            logger.error(
                "KnowledgeStore custom-script: export failed for session %s: %s",
                session.get("id"),
                exc,
                exc_info=True,
            )
