"""Post-split and on-demand session title regeneration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from escriba.app.session_names import strip_part_suffix

if TYPE_CHECKING:
    from escriba.app.database import Database
    from escriba.config import AppConfig

logger = logging.getLogger(__name__)

__all__ = [
    "TitleRegenerationResult",
    "build_title_snippet",
    "regenerate_session_title",
    "strip_part_suffix",
    "title_result_to_dict",
]


@dataclass(frozen=True)
class TitleRegenerationResult:
    """Outcome of a single session title regeneration attempt."""

    session_id: str
    ok: bool
    title: str | None = None
    reason: str | None = None


def title_result_to_dict(result: TitleRegenerationResult) -> dict[str, str | bool]:
    """Serialize a title regeneration result for API responses."""
    payload: dict[str, str | bool] = {"ok": result.ok}
    if result.title:
        payload["title"] = result.title
    if result.reason:
        payload["reason"] = result.reason
    return payload


def build_title_snippet(
    segments: list[dict],
    *,
    max_segments: int = 40,
    max_words: int = 500,
) -> str:
    """Build a transcript snippet suitable for title generation."""
    words = " ".join((segment.get("text") or "") for segment in segments[:max_segments]).split()
    return " ".join(words[:max_words]).strip()


def regenerate_session_title(
    db: Database,
    session_id: str,
    config: AppConfig,
    *,
    app_name: str | None = None,
    generate_title: Callable[..., str | None] | None = None,
) -> TitleRegenerationResult:
    """Generate and persist a descriptive title for a completed session.

    Falls back silently to the existing name when auto-name is disabled or the
    transcript is empty. Failures are logged at warning level and returned in the
    result so callers can surface them to the UI.

    Args:
        db: Session database.
        session_id: Target session id.
        config: Application config (auto-name + summary model).
        app_name: Optional meeting context for the prompt.
        generate_title: Injectable title generator (for tests).

    Returns:
        Structured outcome with the new title when regeneration succeeded.
    """
    if not getattr(config.auto_name, "enabled", True):
        return TitleRegenerationResult(
            session_id,
            ok=False,
            reason="auto_name_disabled",
        )

    segments = db.get_segments(session_id)
    if not segments:
        return TitleRegenerationResult(
            session_id,
            ok=False,
            reason="no_segments",
        )

    snippet = build_title_snippet(
        segments,
        max_words=getattr(config.auto_name, "max_snippet_words", 500),
    )
    if not snippet:
        return TitleRegenerationResult(
            session_id,
            ok=False,
            reason="empty_snippet",
        )

    if generate_title is None:
        from escriba.summarize.llm_summary import generate_session_title

        generate_title = generate_session_title

    try:
        title = generate_title(
            snippet,
            app_name=app_name,
            model=config.streaming.summary_model,
        )
    except Exception as exc:
        logger.warning(
            "Post-split title generation failed for %s: %s",
            session_id,
            exc,
            exc_info=True,
        )
        return TitleRegenerationResult(
            session_id,
            ok=False,
            reason="generation_error",
        )

    if not title:
        logger.warning(
            "Post-split title generation returned no title for %s",
            session_id,
        )
        return TitleRegenerationResult(
            session_id,
            ok=False,
            reason="generation_empty",
        )

    try:
        db.rename_session(session_id, title)
    except Exception as exc:
        logger.warning(
            "Post-split title rename failed for %s: %s",
            session_id,
            exc,
            exc_info=True,
        )
        return TitleRegenerationResult(
            session_id,
            ok=False,
            reason="rename_failed",
        )

    logger.info("Post-split title for %s: %s", session_id, title)
    return TitleRegenerationResult(session_id, ok=True, title=title)
