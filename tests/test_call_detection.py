"""Tests for macOS call detection heuristics."""

from __future__ import annotations

import subprocess

from escriba.audio import call_detection


def test_find_running_meeting_app_checks_all_aliases(monkeypatch):
    """#112: every configured alias is checked, not only process_names[0]."""
    calls: list[str] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0 if cmd[-1] == "zTray" else 1)

    monkeypatch.setattr(call_detection.subprocess, "run", fake_run)

    assert call_detection._find_running_meeting_app() == "zoom"
    assert calls[:3] == ["zoom", "ZoomOpener", "zTray"]


def test_chrome_process_alone_is_not_google_meet(monkeypatch):
    """#112: a generic Chrome process is not enough to count as a Meet call."""

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0 if cmd[-1] == "Google Chrome" else 1)

    monkeypatch.setattr(call_detection.subprocess, "run", fake_run)

    assert call_detection._find_running_meeting_app() is None
