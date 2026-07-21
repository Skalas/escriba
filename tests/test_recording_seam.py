"""Tests for AppState stop-claim / complete_stop_recording seam."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from escriba.app.database import Database
from escriba.app.server import AppState
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
    db = Database(tmp_path / "seam-test.db")
    return AppState(config=minimal_config, db=db)


def test_begin_stop_recording_idempotent_when_not_recording(app_state: AppState) -> None:
    """T3: stop with no active session returns 409 and does not claim."""
    data, status, session = app_state.begin_stop_recording()
    assert status == 409
    assert session is None
    assert data["ok"] is False
    assert app_state._stop_in_progress is False


def test_begin_stop_recording_rejects_second_claim(app_state: AppState) -> None:
    """T3: concurrent second begin_stop while claimed returns 409."""
    fake = MagicMock()
    fake.is_active = True
    fake.db_session_id = "sess-1"
    fake.stop = MagicMock()
    with app_state._lock:
        app_state.session = fake

    data1, status1, session1 = app_state.begin_stop_recording()
    assert status1 == 200
    assert session1 is fake

    data2, status2, session2 = app_state.begin_stop_recording()
    assert status2 == 409
    assert session2 is None
    assert "already" in data2["error"].lower()

    result = app_state.complete_stop_recording(fake)
    assert result["ok"] is True
    assert result["session_id"] == "sess-1"
    fake.stop.assert_called_once()
    assert app_state._stop_in_progress is False


def test_complete_stop_recording_releases_claim_on_stop_error(
    app_state: AppState,
) -> None:
    """T2/T3: claim is released even when session.stop raises."""
    fake = MagicMock()
    fake.is_active = True
    fake.db_session_id = "sess-err"
    fake.stop = MagicMock(side_effect=RuntimeError("boom"))
    with app_state._lock:
        app_state.session = fake

    _data, status, session = app_state.begin_stop_recording()
    assert status == 200 and session is fake
    try:
        app_state.complete_stop_recording(fake)
    except RuntimeError:
        pass
    assert app_state._stop_in_progress is False


def test_try_start_rejected_while_stop_in_progress(app_state: AppState) -> None:
    """T1/T4: starts cannot interleave with an in-flight stop claim."""
    fake = MagicMock()
    fake.is_active = True
    with app_state._lock:
        app_state.session = fake
        app_state._stop_in_progress = True

    payload, status = app_state.try_start_recording()
    assert status == 409
    assert "stop" in payload["error"].lower()


def test_begin_stop_rejected_while_start_in_progress(app_state: AppState) -> None:
    """Stop cannot claim while try_start_recording still holds _start_in_progress."""
    fake = MagicMock()
    fake.is_active = True
    with app_state._lock:
        app_state.session = fake
        app_state._start_in_progress = True

    data, status, session = app_state.begin_stop_recording()
    assert status == 409
    assert session is None
    assert "start" in data["error"].lower()
    assert app_state._stop_in_progress is False


def test_complete_stop_without_claim_raises(app_state: AppState) -> None:
    fake = MagicMock()
    fake.is_active = True
    fake.db_session_id = "x"
    with app_state._lock:
        app_state.session = fake
    try:
        app_state.complete_stop_recording(fake)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "without an active stop claim" in str(exc)


def test_concurrent_stop_claims_only_one_wins(app_state: AppState) -> None:
    """T4: only one of two concurrent begin_stop_recording calls claims."""
    fake = MagicMock()
    fake.is_active = True
    fake.db_session_id = "sess-race"
    fake.stop = MagicMock()
    with app_state._lock:
        app_state.session = fake

    results: list[tuple[dict, int, object]] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait(timeout=5)
        results.append(app_state.begin_stop_recording())

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    statuses = [s for _d, s, _sess in results]
    assert statuses.count(200) == 1
    assert statuses.count(409) == 1
    winner = next(sess for _d, s, sess in results if s == 200)
    app_state.complete_stop_recording(winner)  # type: ignore[arg-type]
