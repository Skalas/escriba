"""Sprint: notes UX, canonical session identity, and knowledge-store sync."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import escriba.app.session as session_module
from escriba.app.database import MERGED_NOTES_SEPARATOR, Database
from escriba.app.server import AppState, _Handler
from escriba.app.session import NOTES_MAX_TOKENS, TranscriptionSession
from escriba.config import AppConfig
from escriba.knowledge.export_worker import run_knowledge_store_export
from escriba.knowledge.local_markdown import LocalMarkdownAdapter
from escriba.transcribe.formats import (
    build_session_export_markdown,
    build_session_export_txt,
)

from .conftest import make_handler

INDEX_HTML = (
    Path(__file__).parent.parent / "src" / "escriba" / "app" / "static" / "index.html"
).read_text(encoding="utf-8")


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "sprint.db")
    yield database
    database.close()


@pytest.fixture
def app_state(db: Database) -> AppState:
    return AppState(config=AppConfig(), db=db)


@pytest.fixture
def handler(app_state: AppState) -> _Handler:
    return make_handler(app_state)


def _completed_session(db: Database, name: str, text: str = "hello") -> str:
    sid = db.create_session(name=name)
    db.add_segments(sid, [{"start": 0.0, "end": 1.0, "text": text, "speaker": "S0"}])
    db.stop_session(sid)
    return sid


def _js_function(name: str) -> str:
    """Return the source of a top-level JS function in index.html."""
    match = re.search(rf"^(?:async )?function {name}\(.*?^\}}", INDEX_HTML, re.S | re.M)
    assert match, f"function {name} not found in index.html"
    return match.group(0)


# --- 1. Canonical session identity -------------------------------------------


def test_get_status_returns_db_session_id_as_canonical(db: Database) -> None:
    sess = TranscriptionSession(AppConfig(), database=db)
    sess.session_id = "20260101_120000"
    sess.db_session_id = "db-uuid"

    status = sess.get_status()

    assert status["session_id"] == "db-uuid"
    assert status["db_session_id"] == "db-uuid"


def test_get_status_falls_back_to_timestamp_id_without_db_row() -> None:
    sess = TranscriptionSession(AppConfig())
    sess.session_id = "20260101_120000"

    assert sess.get_status()["session_id"] == "20260101_120000"
    assert sess.get_status()["db_session_id"] is None


def test_try_start_recording_returns_canonical_session_id(app_state: AppState) -> None:
    class StartedSession:
        canonical_session_id = "db-uuid"

        def __init__(self, config, database=None):
            self.is_active = False
            self.error = None

        def start(self) -> None:
            self.is_active = True

    with patch("escriba.app.session.TranscriptionSession", StartedSession):
        payload, status = app_state.try_start_recording()

    assert status == 200
    assert payload == {"ok": True, "status": "started", "session_id": "db-uuid"}


def test_menubar_auto_start_tracks_canonical_id() -> None:
    source = (
        Path(__file__).parent.parent / "src" / "escriba" / "app" / "menubar.py"
    ).read_text(encoding="utf-8")
    assert "session.session_id" not in source
    assert source.count("canonical_session_id") == 2


# --- 2. Notes state ----------------------------------------------------------


def test_generate_notes_returns_accumulated_notes(db: Database) -> None:
    sid = db.create_session(name="Live")
    db.save_notes(sid, "Earlier notes")
    sess = TranscriptionSession(AppConfig(), database=db)
    sess.db_session_id = sid
    sess.transcriber = MagicMock()
    sess.transcriber.get_full_transcript.return_value = "Transcript"

    with patch.object(
        session_module, "_generate_custom_notes", return_value="New notes"
    ):
        result = sess.generate_notes(prompt="Summarize")

    assert result is not None
    assert result["ok"] is True
    assert result["latest_notes"] == "New notes"
    assert "Earlier notes" in result["notes"]
    assert "New notes" in result["notes"]


def test_generate_notes_reexports_finished_session_only(db: Database) -> None:
    sid = db.create_session(name="Live")
    sess = TranscriptionSession(AppConfig(), database=db)
    sess.db_session_id = sid
    sess.transcriber = MagicMock()
    sess.transcriber.get_full_transcript.return_value = "Transcript"

    with (
        patch.object(session_module, "_generate_custom_notes", return_value="n"),
        patch.object(sess, "_schedule_knowledge_store_export") as schedule,
    ):
        sess.is_active = True
        sess.generate_notes(prompt="p")
        schedule.assert_not_called()
        sess.is_active = False
        sess.generate_notes(prompt="p")
        schedule.assert_called_once()


def test_server_generate_notes_returns_session_result(handler: _Handler) -> None:
    session = MagicMock()
    session.generate_notes.return_value = {
        "ok": True,
        "notes": "all",
        "latest_notes": "new",
    }
    handler.app_state.session = session
    with patch.object(_Handler, "_ensure_local_model_ready", return_value=None):
        payload, status = handler._generate_notes({"prompt": "p", "model": "claude"})

    assert status == 200
    assert payload["notes"] == "all"
    assert payload["latest_notes"] == "new"


def test_live_notepad_is_flushed_before_stop_and_generate() -> None:
    flush = _js_function("flushLiveNotepad")
    assert "clearTimeout(_liveNotepadTimer)" in flush
    assert "await saveLiveNotepad(" in flush

    toggle = _js_function("toggleRecording")
    assert toggle.index("await flushLiveNotepad()") < toggle.index(
        "/api/recording/stop"
    )
    assert "res.session_id" in toggle

    generate = _js_function("generateNotes")
    assert generate.index("await flushLiveNotepad()") < generate.index("'/api/notes'")


def test_reconcile_live_notepad_renders_saved_notes() -> None:
    reconcile = _js_function("reconcileLiveNotepad")
    assert "notes_text" in reconcile
    assert "renderMarkdown(savedNotes)" in reconcile


def test_cancel_notes_edit_restores_saved_ai_notes() -> None:
    cancel = _js_function("cancelNotesEdit")
    assert "_currentSessionNotes" in cancel
    assert "_currentSessionNotes = notes;" in _js_function("saveNotes")
    assert "_currentSessionNotes = s.notes_text" in _js_function("selectSession")


# --- 3. Claude max_tokens ----------------------------------------------------


def test_custom_notes_pass_notes_budget_to_claude() -> None:
    with (
        patch(
            "escriba.summarize.llm_summary.resolve_provider_and_model",
            return_value=("claude", "claude-test"),
        ),
        patch(
            "escriba.summarize.llm_summary._call_llm_claude", return_value="notes"
        ) as call_claude,
    ):
        result = session_module._generate_custom_notes("transcript", "prompt", "claude")

    assert result == "notes"
    assert call_claude.call_args.kwargs["max_tokens"] == NOTES_MAX_TOKENS == 4096


# --- 4. Dashboard / operations -----------------------------------------------


def test_retranscribe_preserves_segments_when_result_empty(
    handler: _Handler, db: Database, tmp_path: Path
) -> None:
    sid = _completed_session(db, "Keep me", text="original")
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\0" * 1024)
    db.update_audio_path(sid, str(wav))

    with patch("escriba.app.session.retranscribe_from_wav", return_value=[]):
        payload, status = handler._retranscribe_session(sid)

    assert status == 500
    assert "existing transcript preserved" in payload["error"]
    assert [s["text"] for s in db.get_segments(sid)] == ["original"]


def test_merge_rejects_duplicate_ids(handler: _Handler, db: Database) -> None:
    sid = _completed_session(db, "A")
    with pytest.raises(ValueError, match="duplicate"):
        db.merge_sessions([sid, sid], "M")
    payload, status = handler._merge_sessions({"session_ids": [sid, sid]})
    assert status == 400
    assert payload["ok"] is False


def test_merge_rejects_active_sessions(handler: _Handler, db: Database) -> None:
    done = _completed_session(db, "Done")
    active = db.create_session(name="Recording")
    payload, status = handler._merge_sessions({"session_ids": [done, active]})
    assert status == 400
    assert "active" in payload["error"]


def test_merge_preserves_notes_and_speaker_labels(db: Database) -> None:
    first = _completed_session(db, "First")
    second = _completed_session(db, "Second")
    db.save_user_notes(first, "mine 1")
    db.save_notes(first, "ai 1")
    db.save_user_notes(second, "mine 2")
    db.set_speaker_label(first, "S0", "Alice")
    db.set_speaker_label(second, "S0", "Bob")
    db.set_speaker_label(second, "S1", "Carol")

    merged_id, _ = db.merge_sessions([second, first], "Merged")
    merged = db.get_session(merged_id)

    assert merged["user_notes"] == f"mine 1{MERGED_NOTES_SEPARATOR}mine 2"
    assert merged["notes_text"] == "ai 1"
    assert db.get_speaker_labels(merged_id) == {"S0": "Alice", "S1": "Carol"}


def test_live_transcript_autoscroll_respects_user_scroll() -> None:
    poll = _js_function("pollTranscript")
    assert "wasNearBottom" in poll
    assert "if (wasNearBottom) box.scrollTop = box.scrollHeight" in poll


def test_sync_status_does_not_hijack_busy_views() -> None:
    sync = _js_function("syncStatus")
    assert "!isUserBusyElsewhere()" in sync
    assert "selectSession(finishedSessionId)" in sync
    busy = _js_function("isUserBusyElsewhere")
    assert "settings-view" in busy and "notes-editor" in busy


def test_view_switches_pause_session_audio() -> None:
    for name in (
        "showLiveView",
        "showEmptyView",
        "showSettingsView",
        "deleteSessionById",
    ):
        assert "pauseSessionAudio()" in _js_function(name), name


def test_speaker_rename_escape_skips_blur_save() -> None:
    rename = _js_function("startSpeakerRename")
    assert "escaped = true" in rename
    assert "if (escaped) return;" in rename


def test_retranscribe_button_keeps_icon() -> None:
    retranscribe = _js_function("retranscribeSession")
    assert ".btn-label" in retranscribe
    assert "btn.innerHTML" not in retranscribe


# --- 5. Knowledge-store export ----------------------------------------------


def test_exports_include_user_notes_before_ai_notes() -> None:
    session = {"name": "S", "user_notes": "mine", "notes_text": "ai"}
    md = build_session_export_markdown(session, [])
    txt = build_session_export_txt(session, [])

    assert md.index("## User Notes") < md.index("## Notes") < md.index("## Transcript")
    assert txt.index("User Notes") < txt.index("\nNotes\n") < txt.index("Transcript")
    assert "## User Notes" not in build_session_export_markdown({"name": "S"}, [])


def test_export_worker_skips_empty_sessions(db: Database) -> None:
    sid = db.create_session(name="Empty")
    adapter = MagicMock()
    with patch("escriba.knowledge.factory.get_knowledge_store", return_value=adapter):
        assert run_knowledge_store_export(db, sid, AppConfig()) is True
    adapter.export.assert_not_called()


def test_export_worker_exports_notes_only_session(db: Database) -> None:
    sid = db.create_session(name="Notes only")
    db.save_user_notes(sid, "jotted")
    adapter = MagicMock()
    with patch("escriba.knowledge.factory.get_knowledge_store", return_value=adapter):
        assert run_knowledge_store_export(db, sid, AppConfig()) is True
    adapter.export.assert_called_once()


def test_local_markdown_removes_stale_export_on_rename(tmp_path: Path) -> None:
    adapter = LocalMarkdownAdapter(output_dir=str(tmp_path))
    session = {"id": "abcdef12-0000", "name": "Session 2026-01-01"}
    unrelated = tmp_path / "Other_99999999.md"
    unrelated.write_text("keep")

    adapter.export(session, None, None, [])
    adapter.export({**session, "name": "Planning"}, None, None, [])

    assert sorted(p.name for p in tmp_path.glob("*.md")) == [
        "Other_99999999.md",
        "Planning_abcdef12.md",
    ]


def test_stop_exports_after_title_refinement(db: Database) -> None:
    sess = TranscriptionSession(AppConfig(), database=db)
    sess.db_session_id = db.create_session(name="s")
    sess.is_active = True
    calls: list[str] = []
    with (
        patch.object(sess, "_join_process_thread", return_value=True),
        patch.object(sess, "_flush_buffer"),
        patch.object(sess, "_close_audio_file"),
        patch.object(sess, "_export"),
        patch.object(sess, "_refine_title", side_effect=lambda: calls.append("refine")),
        patch.object(
            sess,
            "_schedule_knowledge_store_export",
            side_effect=lambda: calls.append("export"),
        ),
    ):
        sess.stop()
    assert calls == ["refine", "export"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda h, sid: h._save_notes(sid, {"notes_text": "n"}),
        lambda h, sid: h._append_notes(sid, {"append_text": "n"}),
        lambda h, sid: h._save_session_user_notes(sid, {"user_notes": "u"}),
        lambda h, sid: h._rename_session(sid, {"name": "Renamed"}),
        lambda h, sid: h._set_speaker_label(sid, {"speaker": "S0", "name": "Ann"}),
    ],
    ids=["notes", "append", "user_notes", "rename", "speaker"],
)
def test_saved_session_mutations_trigger_export(
    handler: _Handler, db: Database, mutate
) -> None:
    sid = _completed_session(db, "Saved")
    with patch.object(_Handler, "_trigger_knowledge_export") as trigger:
        _payload, status = mutate(handler, sid)
    assert status == 200
    trigger.assert_called_once_with(sid)


def test_retranscribe_and_merge_trigger_export(
    handler: _Handler, db: Database, tmp_path: Path
) -> None:
    sid = _completed_session(db, "A")
    other = _completed_session(db, "B")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\0" * 1024)
    db.update_audio_path(sid, str(wav))
    new_segments = [{"start": 0.0, "end": 1.0, "text": "fresh"}]

    with (
        patch.object(_Handler, "_trigger_knowledge_export") as trigger,
        patch("escriba.app.session.retranscribe_from_wav", return_value=new_segments),
        patch.object(_Handler, "_regenerate_title"),
    ):
        assert handler._retranscribe_session(sid)[1] == 200
        payload, status = handler._merge_sessions({"session_ids": [sid, other]})

    assert status == 200
    assert [c.args[0] for c in trigger.call_args_list] == [sid, payload["session_id"]]


def test_trigger_knowledge_export_runs_worker_in_thread(
    handler: _Handler, db: Database
) -> None:
    sid = _completed_session(db, "Saved")
    with patch("escriba.app.server.threading.Thread") as thread_cls:
        handler._trigger_knowledge_export(sid)
    kwargs = thread_cls.call_args.kwargs
    assert kwargs["target"] is run_knowledge_store_export
    assert kwargs["args"] == (db, sid, handler.app_state.config)
    assert kwargs["daemon"] is True
    thread_cls.return_value.start.assert_called_once()


def test_trigger_knowledge_export_skips_active_recording(
    handler: _Handler, db: Database
) -> None:
    active = MagicMock(is_active=True, db_session_id="live")
    handler.app_state.session = active
    with patch("escriba.app.server.threading.Thread") as thread_cls:
        handler._trigger_knowledge_export("live")
    thread_cls.assert_not_called()
