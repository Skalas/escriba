"""Tests for in-browser session export content builders."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from escriba.app.database import Database
from escriba.app.server import (
    AppState,
    _Handler,
    build_session_export_markdown,
    build_session_export_txt,
    format_export_timestamp,
)
from escriba.config import AppConfig
from tests.test_database import _add_segments, _seed_completed_session


@pytest.fixture
def minimal_config(tmp_path: Path) -> AppConfig:
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[audio]
audio_source = "mic"
sample_rate = 16000
channels = 1

[streaming]
backend = "mlx-whisper"
model_size = "tiny"
chunk_duration = 0.5

[auto_name]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    return AppConfig.load(cfg_path)


@pytest.fixture
def app_state(minimal_config: AppConfig, tmp_path: Path) -> AppState:
    db = Database(tmp_path / "export-test.db")
    return AppState(config=minimal_config, db=db)


def _make_handler(app_state: AppState) -> _Handler:
    handler = _Handler.__new__(_Handler)
    handler.app_state = app_state
    handler.headers = {}
    handler.wfile = BytesIO()
    return handler


SAMPLE_SESSION: dict = {
    "name": "Team Sync",
    "started_at": "2026-03-01T10:00:00+00:00",
    "duration_seconds": 125.0,
    "notes_text": "Key decisions made.",
}

SAMPLE_SEGMENTS: list[dict] = [
    {
        "id": 1,
        "start_time": 5.0,
        "text": "Hello everyone",
        "speaker": "SPEAKER_00",
        "speaker_display": "Alice",
    },
    {
        "id": 2,
        "start_time": 65.0,
        "text": "Thanks",
        "speaker": "SPEAKER_01",
        "speaker_display": "Bob",
    },
    {
        "id": 3,
        "start_time": 90.0,
        "text": "No speaker here",
        "speaker": None,
    },
]

EXPECTED_MARKDOWN = "\n".join(
    [
        "# Team Sync",
        "",
        "**Date:** 2026-03-01T10:00:00+00:00",
        "**Duration:** 00:02:05",
        "",
        "## Notes",
        "",
        "Key decisions made.",
        "",
        "## Transcript",
        "",
        '<a id="seg-1"></a>[00:00:05] **Alice**: Hello everyone',
        '<a id="seg-2"></a>[00:01:05] **Bob**: Thanks',
        '<a id="seg-3"></a>[00:01:30] No speaker here',
    ]
)


def test_format_export_timestamp() -> None:
    """Timestamps render as zero-padded HH:MM:SS."""
    assert format_export_timestamp(3661) == "01:01:01"
    assert format_export_timestamp(5) == "00:00:05"


def test_build_session_export_markdown_includes_title_and_transcript() -> None:
    """Markdown export contains title, metadata, and formatted transcript lines."""
    content = build_session_export_markdown(SAMPLE_SESSION, SAMPLE_SEGMENTS)
    assert content == EXPECTED_MARKDOWN


def test_build_session_export_markdown_uses_speaker_display() -> None:
    """Renamed speakers appear in bold on transcript lines."""
    content = build_session_export_markdown(SAMPLE_SESSION, SAMPLE_SEGMENTS)
    assert "**Alice**: Hello everyone" in content
    assert "**Bob**: Thanks" in content
    assert "SPEAKER_00" not in content


def test_build_session_export_markdown_includes_notes_when_present() -> None:
    """Notes section is included when notes_text is set."""
    content = build_session_export_markdown(SAMPLE_SESSION, SAMPLE_SEGMENTS)
    assert "## Notes" in content
    assert "Key decisions made." in content


def test_build_session_export_markdown_omits_notes_when_absent() -> None:
    """Notes section is omitted when notes_text is empty."""
    session = {**SAMPLE_SESSION, "notes_text": "   "}
    content = build_session_export_markdown(session, SAMPLE_SEGMENTS)
    assert "## Notes" not in content


def test_build_session_export_txt_returns_plain_text() -> None:
    """TXT export uses plain formatting without Markdown bold or HTML anchors."""
    content = build_session_export_txt(SAMPLE_SESSION, SAMPLE_SEGMENTS)
    assert content.splitlines()[0] == "Team Sync"
    assert "Date: 2026-03-01T10:00:00+00:00" in content
    assert "Duration: 00:02:05" in content
    assert "Notes" in content
    assert "00:00:05  [Alice] Hello everyone" in content
    assert "**Alice**" not in content
    assert "<a id=" not in content


def test_export_session_missing_returns_404(app_state: AppState) -> None:
    """Export handler returns 404 for unknown sessions."""
    handler = _make_handler(app_state)
    payload, status = handler._export_session("missing-session-id")
    assert status == 404
    assert payload["ok"] is False


def test_export_session_md_from_database(app_state: AppState) -> None:
    """Export handler returns markdown built from stored segments and labels."""
    db = app_state.db
    assert db is not None
    session_id = _seed_completed_session(db, name="Interview", duration=30.0)
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, notes_text = ? WHERE id = ?",
        ("2026-04-01T12:00:00+00:00", "Follow up next week.", session_id),
    )
    db._conn.commit()
    _add_segments(db, session_id, [(10.0, "Tell me about yourself")])
    db._conn.execute(
        "UPDATE segments SET speaker = 'SPEAKER_00' WHERE session_id = ?",
        (session_id,),
    )
    db._conn.commit()
    db.set_speaker_label(session_id, "SPEAKER_00", "Candidate")

    handler = _make_handler(app_state)
    payload, status = handler._export_session(session_id, "md")
    assert status == 200
    assert payload["ok"] is True
    assert payload["filename"].endswith(".md")
    assert "# Interview" in payload["content"]
    assert "**Candidate**: Tell me about yourself" in payload["content"]
    assert "## Notes" in payload["content"]

    txt_payload, txt_status = handler._export_session(session_id, "txt")
    assert txt_status == 200
    assert "[Candidate] Tell me about yourself" in txt_payload["content"]


def test_export_download_returns_raw_attachment(app_state: AppState) -> None:
    """download=1 mode streams raw bytes with Content-Disposition attachment."""
    db = app_state.db
    assert db is not None
    session_id = _seed_completed_session(db, name="Team Sync", duration=30.0)
    _add_segments(db, session_id, [(5.0, "Hello everyone")])

    handler = _make_handler(app_state)
    headers: dict[str, str] = {}
    handler.send_response = MagicMock()
    handler.send_header = lambda key, value: headers.__setitem__(key, value)
    handler.end_headers = MagicMock()

    handler._serve_export_download(session_id, "md")

    handler.send_response.assert_called_once_with(200)
    assert headers["Content-Type"] == "text/markdown; charset=utf-8"
    assert headers["Content-Disposition"] == 'attachment; filename="Team Sync.md"'
    body = handler.wfile.getvalue()
    assert body.startswith(b"# Team Sync")
    assert b'"ok"' not in body


def test_export_download_txt_uses_plain_content_type(app_state: AppState) -> None:
    """TXT download mode uses text/plain and attachment disposition."""
    db = app_state.db
    assert db is not None
    session_id = _seed_completed_session(db, name="Plain Notes", duration=10.0)
    _add_segments(db, session_id, [(1.0, "Line one")])

    handler = _make_handler(app_state)
    headers: dict[str, str] = {}
    handler.send_response = MagicMock()
    handler.send_header = lambda key, value: headers.__setitem__(key, value)
    handler.end_headers = MagicMock()

    handler._serve_export_download(session_id, "txt")

    assert headers["Content-Type"] == "text/plain; charset=utf-8"
    assert headers["Content-Disposition"] == 'attachment; filename="Plain Notes.txt"'
    assert handler.wfile.getvalue().startswith(b"Plain Notes")
