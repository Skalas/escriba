"""Shared session export payload for knowledge-store adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_export_payload(
    session: dict[str, Any],
    summary_json: dict[str, Any] | None,
    audio_path: Path | None,
    segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable export body shared by webhook and custom-script."""
    return {
        "session": session,
        "summary": summary_json,
        "audio_path": str(audio_path) if audio_path else None,
        "segments": segments or [],
    }
