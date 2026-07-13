"""Unit tests for ChunkPump and CaptureSupervisor seams."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from escriba.audio.live_capture import CaptureSupervisor, ChunkPump


def test_chunk_pump_mixed_chunks_equal_duration() -> None:
    """Mixed pop removes equal-duration slices from both streams."""
    pump = ChunkPump(mic_chunk_bytes=4, system_chunk_bytes=8)
    pump.append_system(b"\x01\x02\x03\x04\x05\x06\x07\x08")
    pump.append_mic(b"\xaa\xbb\xcc\xdd")

    mixed = pump.try_pop_mixed_chunks()
    assert mixed == (b"\x01\x02\x03\x04\x05\x06\x07\x08", b"\xaa\xbb\xcc\xdd")
    assert pump.try_pop_mixed_chunks() is None


def test_chunk_pump_mic_only_when_system_empty() -> None:
    """Mic-only path pops when system buffer is empty."""
    pump = ChunkPump(mic_chunk_bytes=3, system_chunk_bytes=6)
    pump.append_mic(b"abcdef")

    assert pump.try_pop_mic_only() == b"abc"
    assert pump.try_pop_mic_only() == b"def"
    assert pump.try_pop_mic_only() is None


def test_chunk_pump_mic_only_retained_when_partial_system_buffered() -> None:
    """Ready mic audio is not dropped when partial system data is still buffering."""
    pump = ChunkPump(mic_chunk_bytes=4, system_chunk_bytes=8)
    pump.append_mic(b"abcdextra")
    pump.append_system(b"\x01\x02")

    assert pump.try_pop_mic_only() is None
    assert pump.try_pop_mic_only() is None

    pump.clear_system()
    assert pump.try_pop_mic_only() == b"abcd"


def test_chunk_pump_system_only_when_mic_empty() -> None:
    """System-only path requires an empty mic buffer."""
    pump = ChunkPump(mic_chunk_bytes=2, system_chunk_bytes=4)
    pump.append_system(b"\x00\x11\x22\x33")

    assert pump.mic_buffer_empty()
    assert pump.try_pop_system_only() == b"\x00\x11\x22\x33"


def test_chunk_pump_pcm_fallback_slices() -> None:
    """Mic-only PCM mode slices at pcm_chunk_bytes."""
    pump = ChunkPump(mic_chunk_bytes=4, system_chunk_bytes=4, pcm_chunk_bytes=5)
    pump.append_pcm(b"1234567890")

    assert pump.try_pop_pcm_chunk() == b"12345"
    assert pump.try_pop_pcm_chunk() == b"67890"
    assert pump.try_pop_pcm_chunk() is None


def test_chunk_pump_clear_system_on_restart() -> None:
    """clear_system drops buffered Swift CLI audio after restart."""
    pump = ChunkPump(mic_chunk_bytes=2, system_chunk_bytes=2)
    pump.append_system(b"\xff\xee")
    pump.clear_system()
    assert not pump.has_system_data()


def test_capture_supervisor_stderr_tail_records_lines() -> None:
    """stderr drain appends trimmed lines to stderr_tail."""
    stop_event = threading.Event()
    supervisor = CaptureSupervisor(stop_event)

    process = MagicMock()
    process.stderr.readline.side_effect = [b"warning: device busy\n", b""]

    supervisor.start_stderr_drain(process)
    time.sleep(0.05)
    stop_event.set()
    supervisor.join_stderr(timeout=1.0)

    assert "warning: device busy" in supervisor.stderr_tail


def test_capture_supervisor_clears_system_buffer_on_restart() -> None:
    """Swift monitor clears system audio via ChunkPump before restart."""
    stop_event = threading.Event()
    supervisor = CaptureSupervisor(stop_event, poll_interval=0.01, max_retries=2)
    pump = ChunkPump(mic_chunk_bytes=2, system_chunk_bytes=2)
    pump.append_system(b"\x01\x02")

    screen_capture = MagicMock()
    screen_capture.process.poll.return_value = 1
    screen_capture.restart.return_value = True
    screen_capture.is_capturing = True

    with patch.object(stop_event, "wait", return_value=False):
        supervisor.start_swift_monitor(screen_capture, pump)
        monitor_thread = supervisor._monitor_thread
        assert monitor_thread is not None
        monitor_thread.join(timeout=2.0)

    stop_event.set()
    assert not pump.has_system_data()
    screen_capture.restart.assert_called()
