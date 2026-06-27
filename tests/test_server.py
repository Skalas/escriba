"""Tests for HTTP server hardening: recording lock, body limits, status codes."""

from __future__ import annotations

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
    _Handler,
)
from escriba.config import AppConfig


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


def _make_handler(app_state: AppState) -> _Handler:
    handler = _Handler.__new__(_Handler)
    handler.app_state = app_state
    handler.headers = {}
    handler.rfile = BytesIO()
    handler.wfile = BytesIO()
    handler.connection = MagicMock()
    return handler


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
