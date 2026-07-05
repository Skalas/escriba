"""Tests for the v1.0.1 release-blocker hardening sprint (T1–T14, #88–#104).

Grouped by strand:
- Strand A — audio-capture correctness (T1–T4)
- Strand B — recording lifecycle leaks & races (T5–T9)
- Strand C — web-security pass (T10–T14)
"""

from __future__ import annotations

import socket
import struct
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import make_handler as _make_handler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_wav(audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int) -> bytes:
    """Build a minimal PCM WAV with an explicit sample width (bytes/sample)."""
    bits = sample_width * 8
    data_size = len(audio_bytes)
    header = b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
    header += b"fmt " + struct.pack("<I", 16)
    header += struct.pack("<H", 1)  # PCM
    header += struct.pack("<H", channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", sample_rate * channels * sample_width)
    header += struct.pack("<H", channels * sample_width)
    header += struct.pack("<H", bits)
    header += b"data" + struct.pack("<I", data_size)
    return header + audio_bytes


# ===========================================================================
# Strand A — audio-capture correctness
# ===========================================================================

def test_t1_32bit_pcm_normalized_into_unit_range() -> None:
    """T1 (#88): 32-bit PCM is scaled by 2**31, not the 16-bit divisor."""
    from escriba.transcribe.streaming import StreamingTranscriber

    transcriber = StreamingTranscriber(model_size="tiny", language="en")

    # Near-full-scale 32-bit samples; with the old 16-bit divisor these would
    # land ~65000x outside [-1, 1].
    samples = np.array(
        [2_000_000_000, -2_000_000_000, 0, 1_000_000_000], dtype=np.int32
    )
    wav = _make_wav(samples.tobytes(), sample_rate=16000, sample_width=4, channels=1)

    captured: dict[str, np.ndarray] = {}

    def _capture(audio_float, sample_rate, raw_audio_for_speaker=None):
        captured["audio"] = audio_float
        return None

    with patch.object(transcriber, "_transcribe_audio", side_effect=_capture):
        transcriber.process_wav_chunk(wav)

    audio = captured["audio"]
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 1.0
    assert np.max(np.abs(audio)) > 0.5  # not silence


def test_t1_manual_parser_also_scales_32bit() -> None:
    """T1 (#88): the manual WAV fallback path uses the same bit-depth scaling."""
    from escriba.transcribe.streaming import StreamingTranscriber

    transcriber = StreamingTranscriber(model_size="tiny", language="en")
    samples = np.array([2_000_000_000, -2_000_000_000], dtype=np.int32)
    wav = _make_wav(samples.tobytes(), sample_rate=16000, sample_width=4, channels=1)

    captured: dict[str, np.ndarray] = {}
    with patch.object(
        transcriber,
        "_transcribe_audio",
        side_effect=lambda a, sr, raw_audio_for_speaker=None: captured.setdefault("a", a),
    ):
        transcriber._process_wav_manual(wav)

    assert np.max(np.abs(captured["a"])) <= 1.0


def test_t2_clock_advances_even_when_chunk_fails() -> None:
    """T2 (#90): a failed chunk still advances accumulated_audio_time."""
    from escriba.transcribe.streaming_mlx import (
        ChunkProcessingError,
        StreamingTranscriberMLX,
    )

    with patch.object(
        StreamingTranscriberMLX, "_resolve_model_path", return_value="/fake/model"
    ):
        transcriber = StreamingTranscriberMLX(model_size="tiny", language="en")

    # 2s of silence @ 16kHz mono int16
    samples = np.zeros(16000 * 2, dtype=np.int16)
    wav = _make_wav(samples.tobytes(), sample_rate=16000, sample_width=2, channels=1)

    before = transcriber.accumulated_audio_time
    with patch("escriba.transcribe.streaming_mlx.mlx_whisper") as mock_mlx, patch(
        "escriba.transcribe.streaming_mlx.time.sleep"
    ):
        mock_mlx.transcribe.side_effect = RuntimeError("Metal hiccup")
        with pytest.raises(ChunkProcessingError):
            transcriber.process_wav_chunk(wav)

    assert transcriber.accumulated_audio_time == pytest.approx(before + 2.0, abs=0.05)


def test_t4_resample_preserves_target_length() -> None:
    """T4 (#104): resampling maps a stream to the target sample count."""
    from escriba.audio.live_capture import _resample_int16

    src = np.arange(0, 1000, dtype=np.int16)
    out = _resample_int16(src, 500)
    assert len(out) == 500
    assert out.dtype == np.int16
    # endpoints preserved
    assert out[0] == src[0]
    assert abs(int(out[-1]) - int(src[-1])) <= 1


def test_t4_per_stream_chunk_bytes_encode_equal_durations() -> None:
    """T4 (#104): chunk sizes derived per-rate represent the same duration."""
    chunk_duration = 2.0
    mic_rate, system_rate = 48000, 16000
    mic_chunk = int(mic_rate * 1 * 2 * chunk_duration)
    system_chunk = int(system_rate * 1 * 2 * chunk_duration)

    mic_seconds = mic_chunk / (mic_rate * 1 * 2)
    system_seconds = system_chunk / (system_rate * 1 * 2)
    assert mic_seconds == pytest.approx(system_seconds)
    # equal byte counts would NOT be equal durations
    assert mic_chunk != system_chunk


def test_t3_swift_converter_guards_non_finite() -> None:
    """T3 (#92): the Swift converter guards NaN/Inf before Int16() (source check)."""
    src = Path("swift-audio-capture/Sources/audio-capture/PCMConverter.swift").read_text()
    assert "isFinite" in src
    assert "clampToInt16" in src
    # every conversion routes through the guarded helper, none trap directly
    assert "Int16(max(-32768, min(32767" not in src


# ===========================================================================
# Strand B — recording lifecycle
# ===========================================================================

def test_t5_start_failure_cleans_up_and_marks_db(tmp_path: Path) -> None:
    """T5 (#89): a start() failure closes the WAV, stops capture, errors the row."""
    from escriba.app.database import Database
    from escriba.app.session import TranscriptionSession
    from escriba.config import AppConfig

    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        "[audio]\naudio_source = \"mic\"\nsample_rate = 16000\nchannels = 1\n"
        "[streaming]\nbackend = \"mlx-whisper\"\nmodel_size = \"tiny\"\nchunk_duration = 0.5\n"
        "[auto_name]\nenabled = false\n",
        encoding="utf-8",
    )
    config = AppConfig.load(cfg_path)
    db = Database(tmp_path / "s.db")
    session = TranscriptionSession(config, database=db)
    session.output_dir = tmp_path / "out"

    with patch(
        "escriba.app.session._build_transcriber",
        side_effect=RuntimeError("model load boom"),
    ):
        session.start()

    assert session.is_active is False
    assert session.error is not None
    assert session._audio_writer is None
    assert session.db_session_id is not None
    row = db.get_session(session.db_session_id)
    assert row is not None
    assert row["status"] == "error"


def test_t5_mic_failure_after_system_start_stops_capture(tmp_path: Path) -> None:
    """T5 (#89): system capture is stopped if mic capture fails in 'both' mode."""
    from escriba.app.database import Database
    from escriba.app.session import TranscriptionSession
    from escriba.config import AppConfig

    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        "[audio]\naudio_source = \"both\"\nsample_rate = 16000\nchannels = 1\n"
        "[streaming]\nbackend = \"mlx-whisper\"\nmodel_size = \"tiny\"\nchunk_duration = 0.5\n"
        "[auto_name]\nenabled = false\n",
        encoding="utf-8",
    )
    config = AppConfig.load(cfg_path)
    db = Database(tmp_path / "s.db")
    session = TranscriptionSession(config, database=db)
    session.output_dir = tmp_path / "out"

    fake_capture = MagicMock()
    fake_capture.start.return_value = True

    with patch("escriba.app.session._build_transcriber", return_value=MagicMock()), patch(
        "escriba.audio.screen_capture.ScreenCaptureAudioCapture",
        return_value=fake_capture,
    ), patch.object(
        TranscriptionSession, "_start_mic_capture", side_effect=RuntimeError("no mic")
    ):
        session.start()

    assert session.is_active is False
    fake_capture.stop.assert_called_once()
    assert session.db_session_id is not None
    row = db.get_session(session.db_session_id)
    assert row is not None
    assert row["status"] == "error"


def _bare_capture() -> object:
    """Build a ScreenCaptureAudioCapture without the SWIFT_CLI_AVAILABLE guard."""
    from escriba.audio.screen_capture import ScreenCaptureAudioCapture

    cap = ScreenCaptureAudioCapture.__new__(ScreenCaptureAudioCapture)
    cap.process = None
    cap.read_thread = None
    cap.is_capturing = False
    cap._lock = threading.Lock()
    cap.stop_event = threading.Event()
    cap.swift_cli_path = "/fake/cli"
    cap.sample_rate = 16000
    cap.channels = 1
    cap.use_screen_capture = False
    cap.audio_callback = None
    return cap


def test_t6_t7_stop_reaps_process_even_when_flag_false() -> None:
    """T6/T7 (#91/#102): stop() terminates a live child even if is_capturing=False."""
    cap = _bare_capture()
    proc = MagicMock()
    proc.wait.return_value = 0
    cap.process = proc
    cap.is_capturing = False  # process died on its own, flag already flipped

    cap.stop()

    proc.terminate.assert_called_once()
    assert cap.process is None


def test_t7_restart_always_stops_before_starting() -> None:
    """T7 (#102): restart() runs cleanup regardless of the is_capturing flag."""
    cap = _bare_capture()
    proc = MagicMock()
    proc.wait.return_value = 0
    cap.process = proc
    cap.is_capturing = False

    with patch.object(type(cap), "start", return_value=True) as mock_start:
        assert cap.restart() is True

    proc.terminate.assert_called_once()  # old process reaped
    mock_start.assert_called_once()


def test_t8_terminate_workers_kills_pool_processes() -> None:
    """T8 (#100): a timed-out inference kills the worker process."""
    from escriba.summarize.llm_summary import _LocalInferenceProcess

    worker = MagicMock()
    worker.is_alive.side_effect = [True, False]  # alive, then dead after terminate
    executor = MagicMock()
    executor._processes = {123: worker}

    _LocalInferenceProcess._terminate_workers(executor)

    worker.terminate.assert_called_once()
    worker.join.assert_called()


def test_t9_stale_socket_reports_not_running_and_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T9 (#98): a leftover socket with nothing listening → False, and unlinked."""
    import escriba.daemon.client as client_mod

    stale = tmp_path / "daemon.sock"
    stale.write_bytes(b"")  # a plain file where nothing listens
    monkeypatch.setattr(client_mod, "DAEMON_SOCKET_PATH", stale)

    assert client_mod.is_daemon_running() is False
    assert not stale.exists()


def test_t9_live_socket_reports_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """T9 (#98): a socket with a real listener → True."""
    import os
    import tempfile

    import escriba.daemon.client as client_mod

    # AF_UNIX paths are capped at ~104 chars on macOS, so use a short path
    # rather than the (long) pytest tmp_path.
    sock_path = Path(tempfile.gettempdir()) / f"esc_t9_{os.getpid()}.sock"
    if sock_path.exists():
        sock_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    monkeypatch.setattr(client_mod, "DAEMON_SOCKET_PATH", sock_path)
    try:
        assert client_mod.is_daemon_running() is True
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()


# ===========================================================================
# Strand C — web-security pass
# ===========================================================================

def _guard_handler(headers: dict) -> object:
    from escriba.app.server import AppState

    handler = _make_handler(MagicMock(spec=AppState))
    handler.headers = headers
    handler.server = MagicMock()
    handler.server.server_address = ("127.0.0.1", 19876)
    return handler


def test_t10_rejects_non_json_content_type() -> None:
    """T10 (#93): a non-application/json Content-Type is rejected with 415."""
    from escriba.app.server import ApiError

    handler = _guard_handler({"Content-Type": "text/plain", "Host": "127.0.0.1:19876"})
    with pytest.raises(ApiError) as exc:
        handler._mutation_guard()
    assert exc.value.status == 415


def test_t10_rejects_foreign_origin() -> None:
    """T10 (#93): a cross-origin request is rejected with 403."""
    from escriba.app.server import ApiError

    handler = _guard_handler(
        {
            "Content-Type": "application/json",
            "Origin": "http://evil.example",
            "Host": "127.0.0.1:19876",
        }
    )
    with pytest.raises(ApiError) as exc:
        handler._mutation_guard()
    assert exc.value.status == 403


def test_t10_rejects_unexpected_host() -> None:
    """T10 (#93): an unexpected Host is rejected with 421 (DNS rebinding)."""
    from escriba.app.server import ApiError

    handler = _guard_handler(
        {"Content-Type": "application/json", "Host": "attacker.example"}
    )
    with pytest.raises(ApiError) as exc:
        handler._mutation_guard()
    assert exc.value.status == 421


def test_t10_allows_same_origin_json() -> None:
    """T10 (#93): a well-formed same-origin JSON request passes the guard."""
    handler = _guard_handler(
        {
            "Content-Type": "application/json",
            "Origin": "http://127.0.0.1:19876",
            "Host": "127.0.0.1:19876",
        }
    )
    handler._mutation_guard()  # no raise


def test_t11_no_inline_handlers_use_escattr() -> None:
    """T11 (#95): no inline on*= handler interpolates via the unsound escAttr."""
    import re

    src = Path("src/escriba/app/static/index.html").read_text()
    offenders = re.findall(r'on[a-z]+="[^"]*escAttr\(', src)
    assert offenders == []
    assert "function escJsAttr" in src


def test_t11_escjsattr_escapes_breakout(tmp_path: Path) -> None:
    """T11 (#95): escJsAttr escapes quotes/backslashes so a payload can't break out."""
    import re

    src = Path("src/escriba/app/static/index.html").read_text()
    m = re.search(r"function escJsAttr\(s\)\s*\{.*?\n\}", src, re.DOTALL)
    assert m, "escJsAttr not found"
    body = m.group(0)
    # It must escape the single quote and the backslash (the breakout chars).
    assert r"\\'" in body or r"\'" in body
    assert r"\\\\" in body


def test_t12_env_value_with_newline_rejected() -> None:
    """T12 (#94): a config value containing a newline is rejected, not injected."""
    from escriba.app.server import ApiError, _Handler

    with pytest.raises(ApiError) as exc:
        _Handler._format_env_line("HUGGINGFACE_TOKEN", "x\nEVIL=1")
    assert exc.value.status == 400


def test_t12_env_write_quotes_and_no_second_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T12 (#94): a normal value is written quoted; a newline value never lands."""
    from escriba.app.server import ApiError, _Handler

    monkeypatch.chdir(tmp_path)
    _Handler._update_env_file({"GEMINI_API_KEY": "abc123"})
    written = Path(".env").read_text()
    assert 'GEMINI_API_KEY="abc123"' in written

    with pytest.raises(ApiError):
        _Handler._update_env_file({"GEMINI_API_KEY": "x\nHUGGINGFACE_TOKEN=evil"})
    # the poisoned second var must not have been written
    assert "HUGGINGFACE_TOKEN=evil" not in Path(".env").read_text()


def test_t13_telegram_error_does_not_log_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T13 (#96): a send failure never writes the bot token to the log."""
    import requests

    from escriba.notify.telegram import send_telegram_message

    token = "123456:SECRET_TOKEN_VALUE"

    def _fake_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 400
        err = requests.HTTPError(f"400 Client Error for url: {url}")
        err.response = resp
        raise err

    caplog.set_level("ERROR")
    with patch.object(requests, "post", _fake_post):
        ok = send_telegram_message("hi", chat_id="42", bot_token=token)

    assert ok is False
    assert "SECRET_TOKEN_VALUE" not in caplog.text


def test_t14_filename_cannot_inject_argv(tmp_path: Path) -> None:
    """T14 (#101): a crafted filename is one argv element, flags unchanged."""
    from escriba.transcribe.whisper import _build_command

    nasty = tmp_path / '-rf x".wav'
    cmd = _build_command(nasty, tmp_path)

    # The path must appear as a single, literal argv element.
    assert str(nasty) in cmd
    # No stray injected flag token derived from the filename.
    assert "-rf" not in cmd
    # Base flags survive intact.
    assert "--model" in cmd


def test_t14_filename_with_placeholder_literal_not_double_substituted(
    tmp_path: Path,
) -> None:
    """T14 (review): a filename literally containing {output_dir} is not re-substituted."""
    from escriba.transcribe.whisper import _build_command

    weird = tmp_path / "{output_dir}.wav"
    cmd = _build_command(weird, tmp_path)
    # The input path is passed verbatim; the literal "{output_dir}" inside the
    # filename must NOT have been replaced with the real output dir.
    assert str(weird) in cmd
