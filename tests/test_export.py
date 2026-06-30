"""Tests for in-browser session export content builders."""

from __future__ import annotations

from pathlib import Path

import pytest

from escriba.app.database import Database
from escriba.app.formats import (
    format_path_for_display,
    save_session_export_to_downloads,
    unique_export_path,
)
from escriba.transcribe.formats import (
    build_session_export_markdown,
    build_session_export_txt,
    format_export_timestamp,
)
from escriba.app.server import AppState
from escriba.config import AppConfig
from tests.conftest import make_handler as _make_handler
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
    assert isinstance(payload["content"], str)
    assert payload["filename"].endswith(".md")
    assert "# Interview" in payload["content"]
    assert "**Candidate**: Tell me about yourself" in payload["content"]
    assert "## Notes" in payload["content"]

    txt_payload, txt_status = handler._export_session(session_id, "txt")
    assert txt_status == 200
    assert "[Candidate] Tell me about yourself" in txt_payload["content"]


def test_save_session_export_writes_file(app_state: AppState, tmp_path: Path) -> None:
    """POST save action writes export to Downloads and returns its path."""
    db = app_state.db
    assert db is not None
    session_id = _seed_completed_session(db, name="Team Sync", duration=30.0)
    _add_segments(db, session_id, [(5.0, "Hello everyone")])

    downloads = tmp_path / "Downloads"
    handler = _make_handler(app_state)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "escriba.app.server.save_session_export_to_downloads",
            lambda content, filename, downloads_dir=None: save_session_export_to_downloads(
                content, filename, downloads_dir=downloads
            ),
        )
        payload, status = handler._save_session_export(session_id, "md")

    assert status == 200
    assert payload["ok"] is True
    saved = Path(payload["path"])
    assert saved.parent == downloads
    assert saved.name == "Team Sync.md"
    assert saved.read_text(encoding="utf-8").startswith("# Team Sync")
    assert payload["display_path"] == format_path_for_display(saved)


def test_save_session_export_deduplicates_filename(
    app_state: AppState, tmp_path: Path
) -> None:
    """Saving twice with the same session name picks a numeric suffix."""
    db = app_state.db
    assert db is not None
    session_id = _seed_completed_session(db, name="Team Sync", duration=30.0)
    _add_segments(db, session_id, [(5.0, "Hello everyone")])

    downloads = tmp_path / "Downloads"
    downloads.mkdir(parents=True)
    (downloads / "Team Sync.md").write_text("existing", encoding="utf-8")

    handler = _make_handler(app_state)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "escriba.app.server.save_session_export_to_downloads",
            lambda content, filename, downloads_dir=None: save_session_export_to_downloads(
                content, filename, downloads_dir=downloads
            ),
        )
        payload, status = handler._save_session_export(session_id, "md")

    assert status == 200
    assert payload["path"].endswith("Team Sync (2).md")
    assert Path(payload["path"]).read_text(encoding="utf-8").startswith("# Team Sync")


def test_save_session_export_missing_returns_404(app_state: AppState) -> None:
    """Save action returns 404 for unknown sessions."""
    handler = _make_handler(app_state)
    payload, status = handler._save_session_export("missing-session-id", "md")
    assert status == 404
    assert payload["ok"] is False


def test_unique_export_path_adds_numeric_suffix(tmp_path: Path) -> None:
    """unique_export_path avoids overwriting an existing file."""
    directory = tmp_path / "Downloads"
    directory.mkdir()
    (directory / "Notes.md").write_text("first", encoding="utf-8")

    second = unique_export_path(directory, "Notes.md")
    assert second.name == "Notes (2).md"
    second.write_text("second", encoding="utf-8")

    third = unique_export_path(directory, "Notes.md")
    assert third.name == "Notes (3).md"
