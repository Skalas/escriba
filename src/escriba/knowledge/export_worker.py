"""Background knowledge-store export worker."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from escriba.app.database import Database
    from escriba.config import AppConfig

logger = logging.getLogger(__name__)


def _has_exportable_content(
    session: dict[str, Any], segments: list[dict[str, Any]]
) -> bool:
    """A session with no transcript and no notes would export an empty stub."""
    if segments:
        return True
    return bool(
        (session.get("user_notes") or "").strip()
        or (session.get("notes_text") or "").strip()
    )


def run_knowledge_store_export(
    db: "Database",
    session_id: str,
    config: "AppConfig",
    audio_path: Path | None = None,
) -> bool:
    """Export a session to the configured knowledge store (fail-soft).

    Returns True when the export ran or was intentionally skipped, False on
    failure. ``audio_path`` defaults to the session's stored audio path.
    """
    try:
        from escriba.knowledge.factory import get_knowledge_store

        session = db.get_session(session_id)
        if not session:
            return True
        segments = db.get_segments(session_id)
        if not _has_exportable_content(session, segments):
            logger.info(
                "Knowledge store export skipped for empty session %s", session_id
            )
            return True
        if audio_path is None and session.get("audio_path"):
            audio_path = Path(session["audio_path"])
        adapter = get_knowledge_store(config.knowledge_store)
        adapter.export(
            session=session,
            summary_json=None,
            audio_path=audio_path,
            segments=segments,
        )
        return True
    except Exception as exc:
        logger.error(
            "Knowledge store export failed for session %s: %s",
            session_id,
            exc,
            exc_info=True,
        )
        return False
