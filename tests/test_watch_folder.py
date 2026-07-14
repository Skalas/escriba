"""Tests for watch-folder processed-set bounds and helpers."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

from escriba.watch import watch_folder as watch_folder_mod
from escriba.watch.watch_folder import (
    DEFAULT_PROCESSED_MAX,
    _BoundedProcessedSet,
    _get_watch_processed_max,
    _process_watch_path,
    _resolve_under_input_dir,
    _retry_transcribe,
)


def test_bounded_processed_set_evicts_oldest_at_cap() -> None:
    bounded = _BoundedProcessedSet(max_size=2)
    first = Path("/tmp/a.wav")
    second = Path("/tmp/b.wav")
    third = Path("/tmp/c.wav")

    assert bounded.add(first) is True
    assert bounded.add(second) is True
    assert first in bounded
    assert second in bounded

    assert bounded.add(third) is True
    assert first not in bounded
    assert second in bounded
    assert third in bounded


def test_bounded_processed_set_rejects_duplicate() -> None:
    bounded = _BoundedProcessedSet(max_size=DEFAULT_PROCESSED_MAX)
    path = Path("/tmp/once.wav")
    assert bounded.add(path) is True
    assert bounded.add(path) is False


def test_bounded_processed_set_discard() -> None:
    bounded = _BoundedProcessedSet(max_size=2)
    path = Path("/tmp/discard.wav")
    bounded.add(path)
    bounded.discard(path)
    assert path not in bounded
    assert bounded.add(path) is True


def test_get_watch_processed_max_invalid_falls_back() -> None:
    with patch.dict(os.environ, {"WATCH_PROCESSED_MAX": "nope"}):
        assert _get_watch_processed_max() == DEFAULT_PROCESSED_MAX
    with patch.dict(os.environ, {"WATCH_PROCESSED_MAX": "0"}):
        assert _get_watch_processed_max() == DEFAULT_PROCESSED_MAX


def test_get_watch_processed_max_valid() -> None:
    with patch.dict(os.environ, {"WATCH_PROCESSED_MAX": "42"}):
        assert _get_watch_processed_max() == 42


def test_resolve_under_input_dir_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "in"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.touch()
    assert _resolve_under_input_dir(outside, root) is None


def test_resolve_under_input_dir_accepts_child(tmp_path: Path) -> None:
    root = tmp_path / "in"
    root.mkdir()
    child = root / "clip.wav"
    child.touch()
    assert _resolve_under_input_dir(child, root) == child.resolve()


def test_active_paths_not_evicted_from_completed(tmp_path: Path) -> None:
    """Completed-set eviction must not drop in-flight/queued dedup keys."""
    completed = _BoundedProcessedSet(max_size=1)
    active: set[Path] = set()
    in_flight = tmp_path / "in_flight.wav"
    done_a = tmp_path / "done_a.wav"
    done_b = tmp_path / "done_b.wav"

    active.add(in_flight)
    completed.add(done_a)
    assert done_a in completed
    assert in_flight not in completed

    completed.add(done_b)
    assert done_a not in completed
    assert done_b in completed
    assert in_flight in active


def test_retry_transcribe_handles_oserror(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wav")
    out = tmp_path / "out"
    with patch(
        "escriba.watch.watch_folder.transcribe_file",
        side_effect=OSError("disk full"),
    ):
        assert _retry_transcribe(audio, out, None, attempts=1, delay=0) is False


def test_process_watch_path_survives_unexpected_error(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wav")
    out = tmp_path / "out"
    with patch(
        "escriba.watch.watch_folder._retry_transcribe",
        side_effect=RuntimeError("unexpected"),
    ):
        assert (
            _process_watch_path(audio, out, None, skip_stability_check=True) is False
        )


def test_worker_survives_unexpected_error_and_processes_next(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    stop_event = threading.Event()
    call_count = {"n": 0}

    def flaky_retry(
        path: Path,
        out: Path,
        combined: Path | None,
        attempts: int = 3,
        delay: float = 1.0,
    ) -> bool:
        call_count["n"] += 1
        if path.name == "first.wav":
            raise RuntimeError("unexpected")
        return True

    def run_watch() -> None:
        with patch.object(watch_folder_mod, "_retry_transcribe", side_effect=flaky_retry):
            watch_folder_mod.watch_folder(
                input_dir,
                output_dir,
                stop_event=stop_event,
                skip_stability_check=True,
            )

    watcher = threading.Thread(target=run_watch, daemon=True)
    watcher.start()
    time.sleep(0.3)
    (input_dir / "first.wav").write_bytes(b"1")
    (input_dir / "second.wav").write_bytes(b"2")
    deadline = time.monotonic() + 10.0
    while call_count["n"] < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    stop_event.set()
    watcher.join(timeout=10.0)

    assert call_count["n"] == 2
    assert not watcher.is_alive()
