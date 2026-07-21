"""T4: app quit must not close the DB while a recording stop is still running."""

from __future__ import annotations

import threading
import types

import pytest

import escriba.app.menubar as menubar
from escriba.app.menubar import TranscriberMenuBar


@pytest.fixture(autouse=True)
def _stub_quit_application(monkeypatch):
    """quit_app ends with rumps.quit_application(); never run the real one in tests."""
    monkeypatch.setattr(menubar.rumps, "quit_application", lambda: None)


class _FakeAppState:
    def __init__(self, session):
        self._lock = threading.Lock()
        self.session = session
        self._stop_in_progress = False
        self._start_in_progress = False

    def begin_stop_recording(self):
        with self._lock:
            if self._stop_in_progress:
                return {"ok": False, "error": "Stop already in progress"}, 409, None
            if self._start_in_progress:
                return {"ok": False, "error": "Recording start in progress"}, 409, None
            session = self.session
            if not session or not session.is_active:
                return {"ok": False, "error": "Not recording"}, 409, None
            self._stop_in_progress = True
            return {"ok": True}, 200, session

    def finish_stop_recording(self) -> None:
        with self._lock:
            self._stop_in_progress = False

    def complete_stop_recording(self, session):
        with self._lock:
            if not self._stop_in_progress:
                raise RuntimeError("complete_stop_recording without claim")
            if self.session is not session:
                self._stop_in_progress = False
                raise RuntimeError("session mismatch")
        try:
            session.stop()
        finally:
            self.finish_stop_recording()
        return {"ok": True, "session_id": getattr(session, "db_session_id", None)}


def _fake_menubar(session, db, server=None):
    """A stand-in with just the attributes quit_app touches (no rumps.App)."""
    self = types.SimpleNamespace()
    self._terminate_dashboard = lambda: None
    self.app_state = _FakeAppState(session)
    self.server = server
    self.db = db
    return self


def test_t4_quit_defers_db_close_while_stop_in_flight(monkeypatch):
    """A stop that outlives the quit timeout must NOT be followed by db.close()."""
    monkeypatch.setattr(menubar, "QUIT_STOP_TIMEOUT_S", 0.05)

    release = threading.Event()

    class SlowSession:
        is_active = True

        def stop(self):
            # Simulate a stop still finalizing the DB row past the quit timeout.
            release.wait(timeout=5)

    class FakeDB:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    db = FakeDB()
    self = _fake_menubar(SlowSession(), db)

    TranscriberMenuBar.quit_app(self, None)

    # Stop is still running → DB must stay open so its pending write survives.
    assert db.closed is False
    release.set()


def test_t4_quit_closes_db_after_stop_completes(monkeypatch):
    """A stop that finishes within the timeout is followed by db.close()."""
    monkeypatch.setattr(menubar, "QUIT_STOP_TIMEOUT_S", 5.0)

    class QuickSession:
        is_active = True

        def stop(self):
            return None

    class FakeDB:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    db = FakeDB()
    self = _fake_menubar(QuickSession(), db)

    TranscriberMenuBar.quit_app(self, None)

    assert db.closed is True


def test_t4_quit_closes_db_when_no_active_session():
    """With no active recording, quit closes the DB immediately."""

    class FakeDB:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    db = FakeDB()
    self = _fake_menubar(None, db)

    TranscriberMenuBar.quit_app(self, None)

    assert db.closed is True
