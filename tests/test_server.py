"""Tests for HTTP server hardening: recording lock, body limits, status codes."""

from __future__ import annotations

import http.client
import json
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from escriba.app.database import Database
from escriba.app.server import (
    MAX_BODY_BYTES,
    ApiError,
    AppState,
    start_server,
)
from escriba.config import AppConfig
from tests.conftest import make_handler as _make_handler


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
    db = Database(tmp_path / "server-test.db")
    return AppState(config=minimal_config, db=db)


def test_t1_concurrent_recording_start_yields_one_active_session(
    app_state: AppState,
) -> None:
    """T1: only one concurrent start succeeds; the other gets 409."""
    created: list[object] = []

    class FakeSession:
        def __init__(self, config, database=None):
            self.config = config
            self.db = database
            self.is_active = False
            self.error = None
            created.append(self)

        def start(self) -> None:
            threading.Event().wait(0.05)
            self.is_active = True

    results: list[tuple[dict, int]] = []

    def worker() -> None:
        results.append(app_state.try_start_recording())

    with patch("escriba.app.session.TranscriptionSession", FakeSession):
        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

    statuses = [status for _payload, status in results]
    assert statuses.count(200) == 1
    assert statuses.count(409) == 1
    assert sum(1 for session in created if session.is_active) == 1
    assert app_state.session is not None
    assert app_state.session.is_active is True


def test_t2_oversized_body_rejected_with_413_before_read(
    app_state: AppState,
) -> None:
    """T2: bodies over ~1MB return 413 and are not read from the socket."""
    handler = _make_handler(app_state)
    handler.headers = {"Content-Length": str(MAX_BODY_BYTES + 1)}
    handler.rfile = MagicMock()

    with pytest.raises(ApiError) as exc_info:
        handler._parse_json_body()

    assert exc_info.value.status == 413
    handler.rfile.read.assert_not_called()


def test_t3_structured_http_status_codes(app_state: AppState) -> None:
    """T3: bad input -> 400, missing resource -> 404, conflict -> 409."""
    handler = _make_handler(app_state)

    handler.headers = {"Content-Length": "12"}
    handler.rfile = BytesIO(b"not-json!!!")
    with pytest.raises(ApiError) as bad_json:
        handler._parse_json_body()
    assert bad_json.value.status == 400

    payload, status = handler._get_session_detail("missing-session-id")
    assert status == 404
    assert payload["ok"] is False

    class ActiveSession:
        def __init__(self, config, database=None):
            self.config = config
            self.db = database
            self.is_active = False
            self.error = None

        def start(self) -> None:
            self.is_active = True

    with patch("escriba.app.session.TranscriptionSession", ActiveSession):
        first = app_state.try_start_recording()
        second = app_state.try_start_recording()

    assert first[1] == 200
    assert second[1] == 409
    assert second[0]["error"] == "Already recording"

    captured: list[tuple[int, dict]] = []

    def capture_json(data: dict, status: int = 200) -> None:
        captured.append((status, data))

    handler._json_response = capture_json  # type: ignore[method-assign]
    handler._respond_error(ApiError("Invalid JSON body", 400))
    assert captured[-1][0] == 400
    assert captured[-1][1] == {"ok": False, "error": "Invalid JSON body"}


def test_json_null_fields_return_4xx_not_500(app_state: AppState) -> None:
    """JSON null for string fields must not raise AttributeError / 500."""
    handler = _make_handler(app_state)

    class NotesSession:
        def generate_notes(
            self, prompt: str | None = None, model: str | None = None
        ) -> None:
            return None

    app_state.session = NotesSession()  # type: ignore[assignment]

    payload, status = handler._generate_notes({"prompt": None})
    assert status == 400
    assert payload["ok"] is False

    payload, status = handler._rename_session("missing-session", {"name": None})
    assert status == 400
    assert payload["ok"] is False

    payload, status = handler._create_folder({"name": None})
    assert status == 400
    assert payload["ok"] is False

    folder_id = app_state.db.create_folder("Existing")
    payload, status = handler._rename_folder(folder_id, {"name": None})
    assert status == 400
    assert payload["ok"] is False


def test_set_speaker_label_rejects_unknown_speaker(app_state: AppState) -> None:
    """Renaming a speaker key that is not in the session returns 400."""
    db = app_state.db
    session_id = db.create_session(name="Interview")
    db.stop_session(session_id)
    db.add_segments(
        session_id,
        [{"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"}],
    )

    handler = _make_handler(app_state)
    payload, status = handler._set_speaker_label(
        session_id,
        {"speaker": "SPEAKER_99", "name": "Ghost"},
    )
    assert status == 400
    assert payload["ok"] is False
    assert payload["error"] == "Unknown speaker for session"
    assert db.get_speaker_labels(session_id) == {}


class _DisconnectWriter:
    """Fake wfile that simulates a client closing the socket mid-write."""

    def write(self, data: bytes) -> int:
        raise BrokenPipeError()

    def flush(self) -> None:
        return None


def test_serve_audio_swallows_client_disconnect(
    app_state: AppState, tmp_path: Path
) -> None:
    """Aborted audio downloads must not raise or trigger error responses."""
    db = app_state.db
    session_id = db.create_session(name="Audio")
    db.stop_session(session_id)
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFF" + b"\x00" * 64)
    db.update_audio_path(session_id, str(wav_path))

    handler = _make_handler(app_state)
    handler.headers = {}
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = _DisconnectWriter()

    handler._serve_audio(session_id)

    handler.send_response.assert_called_once_with(200)


def test_respond_unexpected_swallows_client_disconnect_on_write(
    app_state: AppState, caplog: pytest.LogCaptureFixture
) -> None:
    """Error responses must not double-fault when the client already left."""
    import logging

    caplog.set_level(logging.DEBUG)
    handler = _make_handler(app_state)
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = _DisconnectWriter()

    handler._respond_unexpected(RuntimeError("boom"))

    assert any(
        "Client disconnected during JSON response" in record.getMessage()
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# T4 — TG1: lock-hold latency
# ---------------------------------------------------------------------------

def test_tg1_try_start_recording_releases_lock_before_slow_start(
    app_state: AppState,
) -> None:
    """try_start_recording must not hold app_state._lock while session.start() runs."""
    in_start = threading.Event()
    allow_finish = threading.Event()
    lock_acquired_during_start = threading.Event()

    class SlowSession:
        def __init__(self, config, database=None):
            self.config = config
            self.db = database
            self.is_active = False
            self.error = None

        def start(self) -> None:
            in_start.set()
            allow_finish.wait(timeout=5)
            self.is_active = True

    def _do_start() -> None:
        app_state.try_start_recording()

    def _probe_lock() -> None:
        in_start.wait(timeout=5)
        # The lock must be acquirable while start() is running.
        acquired = app_state._lock.acquire(timeout=1.0)
        if acquired:
            lock_acquired_during_start.set()
            app_state._lock.release()
        allow_finish.set()

    with patch("escriba.app.session.TranscriptionSession", SlowSession):
        t_start = threading.Thread(target=_do_start)
        t_probe = threading.Thread(target=_probe_lock)
        t_start.start()
        t_probe.start()
        t_start.join(timeout=10)
        t_probe.join(timeout=10)

    assert lock_acquired_during_start.is_set(), (
        "app_state._lock was held during session.start(), "
        "which would block /api/status and other reads"
    )


# ---------------------------------------------------------------------------
# T4 — TG2: on-the-wire HTTP dispatch
# ---------------------------------------------------------------------------

@pytest.fixture
def live_server(minimal_config: AppConfig, tmp_path: Path):
    """Start the real HTTP server on an ephemeral port; yield (server, port)."""
    db = Database(tmp_path / "tg2-test.db")
    state = AppState(config=minimal_config, db=db)
    server = start_server(state, port=0)
    port = server.server_address[1]
    yield server, port
    server.shutdown()
    db.close()


def _http(port: int, method: str, path: str, body: bytes = b"", headers: dict | None = None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body or None, headers=headers or {})
    resp = conn.getresponse()
    status = resp.status
    data = json.loads(resp.read())
    conn.close()
    return status, data


def test_tg2_get_status_returns_200_ok(live_server) -> None:
    """GET /api/status over a real socket returns 200 with ok:true."""
    _, port = live_server
    status, body = _http(port, "GET", "/api/status")
    assert status == 200
    assert body["ok"] is True
    assert body["is_active"] is False


def test_tg2_stop_when_not_recording_returns_409(live_server) -> None:
    """POST /api/recording/stop when idle returns 409 with structured error."""
    _, port = live_server
    status, body = _http(port, "POST", "/api/recording/stop")
    assert status == 409
    assert body["ok"] is False
    assert "error" in body


def test_tg2_unknown_session_returns_404(live_server) -> None:
    """GET /api/sessions/<nonexistent> returns 404 with ok:false."""
    _, port = live_server
    status, body = _http(port, "GET", "/api/sessions/no-such-session-xyz")
    assert status == 404
    assert body["ok"] is False


def test_tg2_oversized_content_length_returns_413(live_server) -> None:
    """POST with Content-Length > MAX_BODY_BYTES returns 413 — server checks header before read."""
    _, port = live_server
    # Declare a huge Content-Length but don't send a body; the server rejects
    # on the header check before attempting any body read.
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest("POST", "/api/notes")
    conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
    conn.putheader("Content-Type", "application/json")
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 413
    body = json.loads(resp.read())
    assert body["ok"] is False
    conn.close()


def test_tg2_unknown_route_returns_404(live_server) -> None:
    """GET to an unknown path returns 404."""
    _, port = live_server
    status, body = _http(port, "GET", "/api/does-not-exist")
    assert status == 404
    assert body["ok"] is False


# ---------------------------------------------------------------------------
# T5 — chunked / missing Content-Length body-size cap
# ---------------------------------------------------------------------------

def test_t5_no_content_length_oversized_body_rejected_413(app_state: AppState) -> None:
    """Bodies without Content-Length that exceed the cap must return 413."""
    handler = _make_handler(app_state)
    handler.headers = {}  # no Content-Length header
    handler.rfile = BytesIO(b"x" * (MAX_BODY_BYTES + 1))

    with pytest.raises(ApiError) as exc_info:
        handler._read_body_bytes()

    assert exc_info.value.status == 413


def test_t5_no_content_length_small_body_allowed(app_state: AppState) -> None:
    """A small body without Content-Length is accepted normally."""
    handler = _make_handler(app_state)
    handler.headers = {}
    payload = b'{"key": "value"}'
    handler.rfile = BytesIO(payload)

    result = handler._read_body_bytes()
    assert result == payload


def test_t5_no_content_length_body_at_exact_cap_allowed(app_state: AppState) -> None:
    """A body exactly at the cap (not over) must be accepted."""
    handler = _make_handler(app_state)
    handler.headers = {}
    handler.rfile = BytesIO(b"x" * MAX_BODY_BYTES)

    result = handler._read_body_bytes()
    assert len(result) == MAX_BODY_BYTES


# ---------------------------------------------------------------------------
# T3 — malformed / ambiguous POST path hardening
# ---------------------------------------------------------------------------


def test_t3_post_empty_session_id_returns_404(live_server) -> None:
    """POST /api/sessions//split (empty session_id) must return 404, not 500."""
    _, port = live_server
    status, body = _http(port, "POST", "/api/sessions//split", b"{}")
    assert status == 404
    assert body["ok"] is False


def test_t3_post_extra_path_segments_returns_404(live_server) -> None:
    """POST /api/sessions/<id>/split/extra must return 404, not hit the split handler."""
    _, port = live_server
    status, body = _http(port, "POST", "/api/sessions/123/split/extra", b"{}")
    assert status == 404
    assert body["ok"] is False


def test_t3_post_session_unknown_action_returns_404(live_server) -> None:
    """POST /api/sessions/<id>/unknown-action must return 404."""
    _, port = live_server
    status, body = _http(port, "POST", "/api/sessions/123/unknown-action", b"{}")
    assert status == 404
    assert body["ok"] is False


def test_t3_post_trailing_slash_stripped_correctly(live_server) -> None:
    """Trailing slash on a known POST route must not break routing."""
    _, port = live_server
    # /api/recording/stop/ (trailing slash) should behave like /api/recording/stop
    status, body = _http(port, "POST", "/api/recording/stop/")
    # Not recording → 409; the important thing is it reaches the handler (not 404/500)
    assert status in (409, 200)
    assert body["ok"] is False or body.get("ok") is True


def test_t3_get_empty_session_id_audio_returns_404(live_server) -> None:
    """GET /api/sessions//audio (empty session_id) must return 404."""
    _, port = live_server
    status, body = _http(port, "GET", "/api/sessions//audio")
    assert status == 404
    assert body["ok"] is False


# ---------------------------------------------------------------------------
# T6 — user-notes endpoint
# ---------------------------------------------------------------------------


def test_t6_post_user_notes_persists(live_server) -> None:
    """POST /api/sessions/:id/user-notes must save user_notes and return 200."""
    server, port = live_server
    db: Database = server.RequestHandlerClass.app_state.db  # type: ignore[attr-defined]
    session_id = db.create_session(name="User Notes Test")

    payload = json.dumps({"user_notes": "my context"}).encode()
    status, body = _http(
        port, "POST", f"/api/sessions/{session_id}/user-notes", payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
    )
    assert status == 200
    assert body["ok"] is True

    stored = db.get_session(session_id)
    assert stored is not None
    assert stored["user_notes"] == "my context"


def test_t6_post_user_notes_bad_body_returns_400(live_server) -> None:
    """POST /api/sessions/:id/user-notes with user_notes not a string must return 400."""
    server, port = live_server
    db: Database = server.RequestHandlerClass.app_state.db  # type: ignore[attr-defined]
    session_id = db.create_session(name="Bad Body Test")

    payload = json.dumps({"user_notes": 123}).encode()
    status, body = _http(
        port, "POST", f"/api/sessions/{session_id}/user-notes", payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
    )
    assert status == 400
    assert body["ok"] is False


def test_t6_post_user_notes_null_coerced_to_empty_string(live_server) -> None:
    """POST /api/sessions/:id/user-notes with null user_notes returns 200 and persists ''."""
    server, port = live_server
    db: Database = server.RequestHandlerClass.app_state.db  # type: ignore[attr-defined]
    session_id = db.create_session(name="Null Coerce Test")

    payload = json.dumps({"user_notes": None}).encode()
    status, body = _http(
        port, "POST", f"/api/sessions/{session_id}/user-notes", payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
    )
    assert status == 200
    assert body["ok"] is True

    stored = db.get_session(session_id)
    assert stored is not None
    assert stored["user_notes"] == "", f"Expected empty string, got {stored['user_notes']!r}"


def test_t6_post_user_notes_malformed_path_returns_404(live_server) -> None:
    """POST /api/sessions//user-notes (empty session_id) must return 404."""
    _, port = live_server
    payload = json.dumps({"user_notes": "x"}).encode()
    status, body = _http(
        port, "POST", "/api/sessions//user-notes", payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
    )
    assert status == 404
    assert body["ok"] is False


# ---------------------------------------------------------------------------
# v1.0.0 T4 — /api/version no longer leaks the absolute project_dir path
# ---------------------------------------------------------------------------

def test_t4_version_omits_absolute_project_dir(live_server) -> None:
    """GET /api/version returns build info but not the absolute project_dir path."""
    _, port = live_server
    status, body = _http(port, "GET", "/api/version")
    assert status == 200
    assert body["ok"] is True
    assert "version" in body
    assert "project_dir" not in body


# ---------------------------------------------------------------------------
# v1.0.0 T5 — saved-session generate-notes returns notes; the SPA is the sole
# persister (a server-side save would race the SPA and duplicate/clobber notes).
# ---------------------------------------------------------------------------

def test_t5_generate_notes_returns_without_server_side_persist(app_state: AppState) -> None:
    """Generation returns the notes but must NOT itself write notes_text.

    The SPA combines the generated text with any existing notes and persists it
    (saveNotesForSession / appendNotesToSession). A server-side save here would
    race that path: in the background branch appendNotesToSession reads the value
    back and appends it again, duplicating the notes and clobbering prior ones.
    """
    db = app_state.db
    session_id = db.create_session(name="meeting")
    db.add_segments(session_id, [{"start": 0.0, "end": 1.0, "text": "hello world"}])

    handler = _make_handler(app_state)
    with patch(
        "escriba.app.session._generate_custom_notes",
        return_value="## Summary\n- a point",
    ):
        result, status = handler._generate_session_notes(session_id, {})

    assert status == 200
    assert result["ok"] is True
    assert result["notes"] == "## Summary\n- a point"
    # Server did not persist — notes_text stays empty until the SPA saves.
    reloaded = db.get_session(session_id)
    assert reloaded is not None
    assert not reloaded.get("notes_text")


def test_t5_generate_notes_failure_does_not_persist(app_state: AppState) -> None:
    """A failed generation leaves notes_text untouched."""
    db = app_state.db
    session_id = db.create_session(name="meeting")
    db.add_segments(session_id, [{"start": 0.0, "end": 1.0, "text": "hello world"}])

    handler = _make_handler(app_state)
    with patch("escriba.app.session._generate_custom_notes", return_value=""):
        result, status = handler._generate_session_notes(session_id, {})

    assert status == 503
    assert result["ok"] is False
    reloaded = db.get_session(session_id)
    assert reloaded is not None
    assert not reloaded.get("notes_text")
