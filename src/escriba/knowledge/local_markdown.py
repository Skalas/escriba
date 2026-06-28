"""LocalMarkdownAdapter — writes one .md per session."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

from escriba.knowledge.port import KnowledgeStore

logger = logging.getLogger(__name__)


class LocalMarkdownAdapter(KnowledgeStore):
    """Writes one Markdown file per session into output_dir."""

    def __init__(self, output_dir: str = "~/Documents/Escriba") -> None:
        self._output_dir = Path(output_dir).expanduser()

    def export(
        self,
        session: dict[str, Any],
        summary_json: dict[str, Any] | None,
        audio_path: Path | None,
        segments: list[dict[str, Any]] | None = None,
    ) -> None:
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            filename = self._safe_filename(session)
            path = self._output_dir / filename
            content = self._build_markdown(session, summary_json, segments)
            path.write_text(content, encoding="utf-8")
            logger.info(
                "KnowledgeStore: exported session %s → %s",
                session.get("id"),
                path,
            )
        except Exception as exc:
            logger.error(
                "KnowledgeStore: export failed for session %s: %s",
                session.get("id"),
                exc,
                exc_info=True,
            )

    @staticmethod
    def _safe_filename(session: dict[str, Any]) -> str:
        name = session.get("name") or "Session"
        safe = (
            "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
            or "Session"
        )
        session_id = (session.get("id") or "")[:8]
        return f"{safe}_{session_id}.md"

    @staticmethod
    def _build_markdown(
        session: dict[str, Any],
        summary_json: dict[str, Any] | None,
        segments: list[dict[str, Any]] | None = None,
    ) -> str:
        from escriba.app.server import build_session_export_markdown

        lines: list[str] = []
        md = build_session_export_markdown(session, segments or [])
        lines.append(md)
        if summary_json:
            lines.append("\n## AI Summary\n")
            for key, val in summary_json.items():
                if isinstance(val, list) and val:
                    lines.append(f"\n### {key.replace('_', ' ').title()}\n")
                    for item in val:
                        if isinstance(item, dict):
                            task = item.get("task", "")
                            assignee = item.get("assignee", "")
                            line = f"- {task}"
                            if assignee:
                                line += f" — {assignee}"
                            lines.append(line)
                        else:
                            lines.append(f"- {item}")
                elif isinstance(val, str) and val:
                    lines.append(f"\n### {key.replace('_', ' ').title()}\n\n{val}")
        return "\n".join(lines)
