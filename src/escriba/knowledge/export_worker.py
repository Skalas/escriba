"""Background knowledge-store export worker."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escriba.app.database import Database
    from escriba.config import AppConfig

logger = logging.getLogger(__name__)


def run_knowledge_store_export(
    db: "Database",
    session_id: str,
    config: "AppConfig",
    audio_path: Path | None,
) -> None:
    """Export a completed session to the configured knowledge store (fail-soft)."""
    try:
        from escriba.knowledge.factory import get_knowledge_store

        session = db.get_session(session_id)
        if not session:
            return
        segments = db.get_segments(session_id)
        adapter = get_knowledge_store(config.knowledge_store)
        adapter.export(
            session=session,
            summary_json=None,
            audio_path=audio_path,
            segments=segments,
        )
    except Exception as exc:
        logger.error(
            "Knowledge store export failed for session %s: %s",
            session_id,
            exc,
            exc_info=True,
        )
