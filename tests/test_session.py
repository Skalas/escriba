"""Tests for transcription session lifecycle and LLM serialization."""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from escriba.app.database import Database
from escriba.app.session import TranscriptionSession
from escriba.config import AppConfig
from escriba.summarize import llm_summary


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
def mocked_session(
    minimal_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TranscriptionSession:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db = Database(tmp_path / "session-test.db")
    mock_transcriber = MagicMock()
    mock_transcriber.get_full_transcript.return_value = ""
    mock_transcriber.segments = []
    mock_transcriber.lock = threading.Lock()
    mock_transcriber.export_transcript.return_value = None

    with patch(
        "escriba.transcribe.streaming_mlx.StreamingTranscriberMLX",
        return_value=mock_transcriber,
    ), patch.object(TranscriptionSession, "_start_mic_capture"), patch.object(
        TranscriptionSession, "_refine_title"
    ), patch.object(TranscriptionSession, "_export"):
        session = TranscriptionSession(minimal_config, database=db)
        session.start()
        yield session
        if session.is_active:
            session.stop()
        db.close()


def test_t7_session_lifecycle_start_stop(mocked_session: TranscriptionSession) -> None:
    """T7: start activates the session; stop completes it in the DB."""
    session = mocked_session
    assert session.is_active is True
    assert session.db_session_id is not None

    db_row = session.db.get_session(session.db_session_id)
    assert db_row is not None
    assert db_row["status"] == "active"

    session.stop()

    assert session.is_active is False
    db_row = session.db.get_session(session.db_session_id)
    assert db_row is not None
    assert db_row["status"] == "completed"


def test_t8_audio_wav_persisted_on_stop(
    minimal_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T8: stopping a session writes a WAV file and stores its path."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db = Database(tmp_path / "audio-test.db")
    mock_transcriber = MagicMock()
    mock_transcriber.get_full_transcript.return_value = ""
    mock_transcriber.segments = []
    mock_transcriber.lock = threading.Lock()

    try:
        with patch(
            "escriba.transcribe.streaming_mlx.StreamingTranscriberMLX",
            return_value=mock_transcriber,
        ), patch.object(TranscriptionSession, "_start_mic_capture"), patch.object(
            TranscriptionSession, "_refine_title"
        ), patch.object(TranscriptionSession, "_export"):
            session = TranscriptionSession(minimal_config, database=db)
            session.start()
            assert session._audio_writer is not None
            session._audio_writer.writeframes(b"\x00\x00" * 800)
            session.stop()

        assert session._audio_file is not None
        assert session._audio_file.exists()

        with wave.open(str(session._audio_file), "rb") as wf:
            assert wf.getnframes() > 0

        assert session.db_session_id is not None
        db_row = db.get_session(session.db_session_id)
        assert db_row is not None
        assert db_row["audio_path"] == str(session._audio_file)
    finally:
        db.close()


def test_t9_local_llm_calls_are_serialized() -> None:
    """T9: the mlx-lm semaphore allows at most one generation at a time."""
    active = 0
    peak = 0
    lock = threading.Lock()
    first_entered = threading.Event()
    release = threading.Event()

    def tracked_run(*_args, **_kwargs) -> str:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        first_entered.set()
        release.wait(timeout=5)
        with lock:
            active -= 1
        return "ok"

    def worker(results: list[str | None]) -> None:
        results.append(
            llm_summary._call_llm_local("prompt", "test-model", max_tokens=8)
        )

    with patch(
        "escriba.summarize.llm_summary._run_local_generation",
        side_effect=tracked_run,
    ):
        results: list[str | None] = []
        t1 = threading.Thread(target=worker, args=(results,))
        t2 = threading.Thread(target=worker, args=(results,))
        t1.start()
        assert first_entered.wait(timeout=2)
        with lock:
            assert active == 1

        t2.start()
        threading.Event().wait(0.1)
        with lock:
            assert active == 1

        release.set()
        t1.join(timeout=5)
        release.set()
        t2.join(timeout=5)

    assert peak == 1
    assert results == ["ok", "ok"]
