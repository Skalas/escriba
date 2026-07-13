"""Tests for post-split session title regeneration (strand A / T1–T4)."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from escriba.app.database import Database
from escriba.app.server import AppState
from escriba.app.session_titles import (
    regenerate_session_title,
    strip_part_suffix,
    title_result_to_dict,
)
from escriba.config import AppConfig
from tests.conftest import make_handler as _make_handler
from tests.test_database import _add_segments, _seed_completed_session


def _session_name(db: Database, session_id: str) -> str:
    """Return session name, asserting the row exists (for mypy)."""
    session = db.get_session(session_id)
    assert session is not None
    return str(session["name"])


@pytest.fixture
def auto_name_config(tmp_path: Path) -> AppConfig:
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
summary_model = "gemini"

[auto_name]
enabled = true
min_segments = 1
max_snippet_words = 500
""".strip(),
        encoding="utf-8",
    )
    return AppConfig.load(cfg_path)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "split-title.db")
    yield database
    database.close()


def test_strip_part_suffix_removes_nested_part_label() -> None:
    assert strip_part_suffix("Weekly sync (part 1)") == "Weekly sync"
    assert strip_part_suffix("Weekly sync (part 2)") == "Weekly sync"


def test_regenerate_session_title_success(db: Database, auto_name_config: AppConfig) -> None:
    session_id = _seed_completed_session(db, name="Meeting (part 1)")
    _add_segments(
        db,
        session_id,
        [(0.0, "We reviewed the quarterly roadmap"), (5.0, "Budget is on track")],
    )

    result = regenerate_session_title(
        db,
        session_id,
        auto_name_config,
        generate_title=lambda _snippet, **_kwargs: "Quarterly Roadmap Review",
    )

    assert result.ok is True
    assert result.title == "Quarterly Roadmap Review"
    assert _session_name(db, session_id) == "Quarterly Roadmap Review"


def test_regenerate_session_title_failure_keeps_part_name(
    db: Database,
    auto_name_config: AppConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = _seed_completed_session(db, name="Meeting (part 2)")
    _add_segments(db, session_id, [(0.0, "Follow-up on hiring plan")])

    with caplog.at_level(logging.WARNING):
        result = regenerate_session_title(
            db,
            session_id,
            auto_name_config,
            generate_title=lambda _snippet, **_kwargs: None,
        )

    assert result.ok is False
    assert result.reason == "generation_empty"
    assert _session_name(db, session_id) == "Meeting (part 2)"
    assert any("returned no title" in record.message for record in caplog.records)


def test_regenerate_session_title_disabled_skips_rename(
    db: Database,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "disabled-auto-name.toml"
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
summary_model = "gemini"

[auto_name]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    config = AppConfig.load(cfg_path)
    session_id = _seed_completed_session(db, name="Meeting (part 1)")
    _add_segments(db, session_id, [(0.0, "Hello world")])

    result = regenerate_session_title(db, session_id, config)

    assert result.ok is False
    assert result.reason == "auto_name_disabled"
    assert _session_name(db, session_id) == "Meeting (part 1)"


def test_split_session_returns_title_status_for_both_halves(
    db: Database,
    auto_name_config: AppConfig,
) -> None:
    session_id = _seed_completed_session(db, name="Long meeting", duration=120.0)
    seg_ids = _add_segments(
        db,
        session_id,
        [(0.0, "Morning standup topics"), (30.0, "Architecture review")],
    )
    app_state = AppState(config=auto_name_config, db=db)
    handler = _make_handler(app_state)

    def fake_generate(snippet: str, **_kwargs: str | None) -> str | None:
        if "standup" in snippet.lower():
            return "Morning Standup"
        if "architecture" in snippet.lower():
            return "Architecture Review"
        return None

    with patch(
        "escriba.summarize.llm_summary.generate_session_title",
        side_effect=fake_generate,
    ):
        payload, status = handler._split_session(
            session_id,
            {"segment_id": seg_ids[1]},
        )

    assert status == 200
    assert payload["ok"] is True
    assert payload["titles"]["first"]["ok"] is True
    assert payload["titles"]["second"]["ok"] is True
    assert _session_name(db, session_id) == "Morning Standup"
    assert _session_name(db, payload["second_session_id"]) == "Architecture Review"


def test_manual_rename_after_split_persists(
    db: Database,
    auto_name_config: AppConfig,
) -> None:
    session_id = _seed_completed_session(db, name="Interview", duration=60.0)
    seg_ids = _add_segments(
        db,
        session_id,
        [(0.0, "Tell me about yourself"), (20.0, "System design question")],
    )
    app_state = AppState(config=auto_name_config, db=db)
    handler = _make_handler(app_state)

    with patch(
        "escriba.summarize.llm_summary.generate_session_title",
        return_value=None,
    ):
        split_payload, _ = handler._split_session(
            session_id,
            {"segment_id": seg_ids[1]},
        )

    second_id = split_payload["second_session_id"]
    rename_payload, rename_status = handler._rename_session(
        second_id,
        {"name": "System Design Interview"},
    )

    assert rename_status == 200
    assert rename_payload["ok"] is True
    assert _session_name(db, second_id) == "System Design Interview"


def test_regenerate_session_title_rename_failure_returns_structured_result(
    db: Database,
    auto_name_config: AppConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = _seed_completed_session(db, name="Meeting (part 1)")
    _add_segments(db, session_id, [(0.0, "Budget review")])

    db.rename_session = MagicMock(side_effect=sqlite3.OperationalError("locked"))  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING):
        result = regenerate_session_title(
            db,
            session_id,
            auto_name_config,
            generate_title=lambda _snippet, **_kwargs: "Budget Review",
        )

    assert result.ok is False
    assert result.reason == "rename_failed"
    assert _session_name(db, session_id) == "Meeting (part 1)"
    assert any("rename failed" in record.message.lower() for record in caplog.records)


def test_split_session_survives_rename_failure(
    db: Database,
    auto_name_config: AppConfig,
) -> None:
    session_id = _seed_completed_session(db, name="Long meeting", duration=120.0)
    seg_ids = _add_segments(
        db,
        session_id,
        [(0.0, "Morning standup topics"), (30.0, "Architecture review")],
    )
    app_state = AppState(config=auto_name_config, db=db)
    handler = _make_handler(app_state)
    original_rename = db.rename_session
    calls = {"count": 0}

    def flaky_rename(sid: str, name: str) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise sqlite3.OperationalError("forced rename failure")
        original_rename(sid, name)

    db.rename_session = flaky_rename  # type: ignore[method-assign]

    with patch(
        "escriba.summarize.llm_summary.generate_session_title",
        side_effect=lambda snippet, **_kwargs: (
            "Morning Standup" if "standup" in snippet.lower() else "Architecture Review"
        ),
    ):
        payload, status = handler._split_session(
            session_id,
            {"segment_id": seg_ids[1]},
        )

    assert status == 200
    assert payload["ok"] is True
    assert payload["titles"]["first"]["ok"] is False
    assert payload["titles"]["first"]["reason"] == "rename_failed"
    assert payload["titles"]["second"]["ok"] is True
    assert _session_name(db, session_id) == "Long meeting (part 1)"
    assert _session_name(db, payload["second_session_id"]) == "Architecture Review"


def test_title_result_to_dict_includes_reason() -> None:
    from escriba.app.session_titles import TitleRegenerationResult

    payload = title_result_to_dict(
        TitleRegenerationResult("sid", ok=False, reason="generation_error")
    )
    assert payload == {"ok": False, "reason": "generation_error"}
