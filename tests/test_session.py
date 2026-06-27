"""Tests for transcription session lifecycle and LLM serialization."""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from escriba.app.database import Database
from escriba.app.session import (
    TranscriptionSession,
    _summary_to_markdown,
)
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


def test_summary_to_markdown_renders_full_summary() -> None:
    """Default notes path returns markdown sections, not raw JSON."""
    result = {
        "summary": "Team aligned on the release plan.",
        "key_points": ["Scope is frozen", "QA starts Monday"],
        "action_items": [
            {"task": "Send recap", "assignee": "Alice", "due_date": "Friday"},
            {"task": "Review spec", "assignee": "", "due_date": ""},
            {"task": "Ping vendor", "assignee": "Bob", "due_date": ""},
        ],
        "decisions": ["Ship v0.4.0 next week"],
        "topics": ["Release", "QA"],
    }

    markdown = _summary_to_markdown(result)

    assert "## Summary" in markdown
    assert "Team aligned on the release plan." in markdown
    assert "## Key Points" in markdown
    assert "- Scope is frozen" in markdown
    assert "## Action Items" in markdown
    assert "- Send recap — Alice (due: Friday)" in markdown
    assert "- Review spec" in markdown
    assert "Review spec —" not in markdown
    assert "- Ping vendor — Bob" in markdown
    assert "(due: )" not in markdown
    assert "## Decisions" in markdown
    assert "- Ship v0.4.0 next week" in markdown
    assert "## Topics" in markdown
    assert "- Release" in markdown
    assert '"summary"' not in markdown
    assert "{" not in markdown


def test_summary_to_markdown_omits_empty_sections() -> None:
    """Sections with no content are left out of the markdown."""
    markdown = _summary_to_markdown(
        {
            "summary": "Brief recap only.",
            "key_points": [],
            "action_items": [],
            "decisions": [],
            "topics": [],
        }
    )

    assert "## Summary" in markdown
    assert "Brief recap only." in markdown
    assert "## Key Points" not in markdown
    assert "## Action Items" not in markdown
    assert "## Decisions" not in markdown
    assert "## Topics" not in markdown


def test_t3_audio_buffer_backpressure_drops_oldest_and_warns(
    minimal_config: AppConfig,
) -> None:
    """T3: live PCM buffer stays capped when transcription falls behind."""
    session = TranscriptionSession(minimal_config)
    chunk_bytes = session._chunk_pcm_byte_size()
    cap = session._audio_buffer_cap_bytes()

    with patch.object(session, "_last_buffer_overflow_log", 0.0):
        with patch("escriba.app.session.logger") as mock_logger:
            session._on_audio_data(b"\x01" * (cap - 100))
            assert len(session._audio_buffer) == cap - 100

            session._on_audio_data(b"\x02" * chunk_bytes)
            assert len(session._audio_buffer) <= cap
            mock_logger.warning.assert_called_once()

            session._on_audio_data(b"\x03" * chunk_bytes)
            assert len(session._audio_buffer) <= cap
            mock_logger.warning.assert_called_once()


def test_t3_audio_buffer_under_cap_unchanged(minimal_config: AppConfig) -> None:
    """T3: normal ingestion below the cap keeps all buffered PCM."""
    session = TranscriptionSession(minimal_config)
    cap = session._audio_buffer_cap_bytes()
    data = b"\x02\x00" * 100

    session._on_audio_data(data)
    session._on_audio_data(data)

    assert len(session._audio_buffer) == len(data) * 2
    assert len(session._audio_buffer) <= cap


@pytest.fixture
def distinctive_config(minimal_config: AppConfig) -> AppConfig:
    from dataclasses import replace

    from escriba.config import DictionaryConfig
    from escriba.transcribe.config import HallucinationConfig, VADConfig

    return replace(
        minimal_config,
        vad=VADConfig(threshold=0.42, min_silence_duration_ms=777),
        hallucination=HallucinationConfig(
            condition_on_previous_text=True,
            no_speech_threshold=0.33,
            compression_ratio_threshold=1.8,
            logprob_threshold=-0.5,
        ),
        dictionary=DictionaryConfig(
            terms=["Escriba"],
            replacements={"acme": "ACME Corp"},
        ),
        streaming=replace(
            minimal_config.streaming,
            vad_enabled=False,
            backend="mlx-whisper",
            device="cpu",
        ),
    )


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    from escriba.app.session import _build_wav

    pcm = b"\x00\x00" * 16000
    wav_path = tmp_path / "retranscribe.wav"
    wav_path.write_bytes(_build_wav(pcm, sample_rate=16000, channels=1))
    return wav_path


def _mock_transcriber() -> MagicMock:
    mock = MagicMock()
    mock.segments = []
    mock.process_wav_chunk.return_value = None
    return mock


def test_t1_retranscribe_mlx_passes_full_streaming_config(
    distinctive_config: AppConfig, wav_file: Path
) -> None:
    """T1: retranscribe_from_wav mirrors live MLX config (VAD, hallucination, dictionary)."""
    from escriba.app.session import retranscribe_from_wav

    mock_transcriber = _mock_transcriber()
    with patch(
        "escriba.transcribe.streaming_mlx.StreamingTranscriberMLX",
        return_value=mock_transcriber,
    ) as mock_cls:
        retranscribe_from_wav(wav_file, distinctive_config)

    mock_cls.assert_called_once_with(
        model_size=distinctive_config.streaming.model_size,
        language=distinctive_config.streaming.language,
        realtime_output=False,
        vad_enabled=distinctive_config.streaming.vad_enabled,
        vad_config=distinctive_config.vad,
        hallucination_config=distinctive_config.hallucination,
        dictionary=distinctive_config.dictionary,
    )


def test_t1_retranscribe_faster_whisper_passes_full_streaming_config(
    distinctive_config: AppConfig, wav_file: Path
) -> None:
    """T1: retranscribe_from_wav mirrors live faster-whisper config."""
    from dataclasses import replace

    from escriba.app.session import retranscribe_from_wav

    config = replace(
        distinctive_config,
        streaming=replace(
            distinctive_config.streaming,
            backend="faster-whisper",
            device="cpu",
        ),
    )
    mock_transcriber = _mock_transcriber()
    with patch(
        "escriba.transcribe.streaming.StreamingTranscriber",
        return_value=mock_transcriber,
    ) as mock_cls:
        retranscribe_from_wav(wav_file, config)

    mock_cls.assert_called_once_with(
        model_size=config.streaming.model_size,
        language=config.streaming.language,
        device=config.streaming.device,
        realtime_output=False,
        vad_enabled=config.streaming.vad_enabled,
        vad_config=config.vad,
        hallucination_config=config.hallucination,
    )
