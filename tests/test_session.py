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


def test_t9_local_llm_calls_use_subprocess_worker() -> None:
    """T9: _call_llm_local delegates to _LocalInferenceProcess which uses max_workers=1.

    The subprocess pool (max_workers=1) is the serialization mechanism.
    We verify: mlx_lm absent → early exit; mlx_lm present → delegates to
    the process worker and returns whatever it returns.
    """
    import sys

    # When mlx_lm is not importable, _call_llm_local returns None immediately.
    saved = sys.modules.pop("mlx_lm", None)
    try:
        result = llm_summary._call_llm_local("prompt", "model", max_tokens=8)
        assert result is None
    finally:
        if saved is not None:
            sys.modules["mlx_lm"] = saved

    # When mlx_lm is available, delegates to _local_inference_process.run
    calls: list[tuple] = []

    def capture_run(prompt, model_id, max_tokens, enable_thinking):
        calls.append((prompt, model_id, max_tokens, enable_thinking))
        return "generated"

    with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
        with patch.object(llm_summary._local_inference_process, "run", side_effect=capture_run):
            result = llm_summary._call_llm_local("p", "m", max_tokens=32, enable_thinking=False)

    assert result == "generated"
    assert calls == [("p", "m", 32, False)]

    # Verify the process pool is configured with max_workers=1 (the serialization guarantee).

    proc = llm_summary._LocalInferenceProcess()
    executor = proc._get_executor()
    try:
        assert executor._max_workers == 1
    finally:
        executor.shutdown(wait=False)


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


def test_both_mode_mix_aligns_unequal_buffers_to_trailing_window(
    tmp_path: Path,
) -> None:
    """Asymmetric backpressure must not time-shift one source via trailing silence pad."""
    import numpy as np

    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[audio]
audio_source = "both"
sample_rate = 16000
channels = 1
mic_boost = 1.0

[streaming]
chunk_duration = 0.5

[auto_name]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    config = AppConfig.load(cfg_path)
    session = TranscriptionSession(config)

    # System kept an extra leading frame; mic backpressure trimmed more aggressively.
    session._system_buffer = bytearray(b"\x0a\x00\x14\x00\x1e\x00")  # 10, 20, 30
    session._mic_buffer = bytearray(b"\x32\x00\x3c\x00")  # 50, 60

    with session._buffer_lock:
        mixed = session._mix_buffers()

    assert len(session._system_buffer) == len(session._mic_buffer) == 4
    assert len(mixed) == 4

    sys_tail = np.frombuffer(bytes(session._system_buffer), dtype=np.int16)
    mic_tail = np.frombuffer(bytes(session._mic_buffer), dtype=np.int16)
    mixed_samples = np.frombuffer(mixed, dtype=np.int16)

    assert sys_tail.tolist() == [20, 30]
    assert mic_tail.tolist() == [50, 60]
    np.testing.assert_array_equal(mixed_samples, sys_tail + mic_tail)


def test_both_mode_mix_preserves_system_when_mic_empty(tmp_path: Path) -> None:
    """Empty mic must not trim away system audio in 'both' mode."""
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[audio]
audio_source = "both"
sample_rate = 16000
channels = 1

[streaming]
chunk_duration = 0.5

[auto_name]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    session = TranscriptionSession(AppConfig.load(cfg_path))
    system_pcm = b"\x0a\x00\x14\x00\x1e\x00"
    session._system_buffer = bytearray(system_pcm)
    session._mic_buffer = bytearray()

    with session._buffer_lock:
        mixed = session._mix_buffers()

    assert mixed == system_pcm
    assert bytes(session._system_buffer) == system_pcm


def test_both_mode_mix_preserves_mic_when_system_empty(tmp_path: Path) -> None:
    """Empty system audio must not trim away mic audio in 'both' mode."""
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[audio]
audio_source = "both"
sample_rate = 16000
channels = 1

[streaming]
chunk_duration = 0.5

[auto_name]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    session = TranscriptionSession(AppConfig.load(cfg_path))
    mic_pcm = b"\x32\x00\x3c\x00"
    session._system_buffer = bytearray()
    session._mic_buffer = bytearray(mic_pcm)

    with session._buffer_lock:
        mixed = session._mix_buffers()

    assert mixed == mic_pcm
    assert bytes(session._mic_buffer) == mic_pcm


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
    """T1/B2: retranscribe uses _build_transcriber with full MLX config."""
    from escriba.app.session import retranscribe_from_wav

    mock_transcriber = _mock_transcriber()
    with patch(
        "escriba.app.session._build_transcriber",
        return_value=mock_transcriber,
    ) as mock_build:
        retranscribe_from_wav(wav_file, distinctive_config)

    mock_build.assert_called_once_with(
        distinctive_config,
        realtime_output=False,
    )


def test_b2_build_transcriber_mlx_passes_full_streaming_config(
    distinctive_config: AppConfig,
) -> None:
    """B2: shared helper passes VAD, hallucination, and dictionary to MLX."""
    from escriba.app.session import _build_transcriber

    mock_transcriber = _mock_transcriber()
    with patch(
        "escriba.transcribe.streaming_mlx.StreamingTranscriberMLX",
        return_value=mock_transcriber,
    ) as mock_cls:
        _build_transcriber(distinctive_config, realtime_output=True)

    mock_cls.assert_called_once_with(
        model_size=distinctive_config.streaming.model_size,
        language=distinctive_config.streaming.language,
        realtime_output=True,
        vad_enabled=distinctive_config.streaming.vad_enabled,
        vad_config=distinctive_config.vad,
        hallucination_config=distinctive_config.hallucination,
        dictionary=distinctive_config.dictionary,
    )


def test_t1_retranscribe_faster_whisper_passes_full_streaming_config(
    distinctive_config: AppConfig, wav_file: Path
) -> None:
    """T1/B2: _build_transcriber honors faster-whisper backend selection."""
    from dataclasses import replace

    from escriba.app.session import _build_transcriber, retranscribe_from_wav

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
        realtime_output=False,
        vad_enabled=config.streaming.vad_enabled,
        vad_config=config.vad,
        hallucination_config=config.hallucination,
        device=config.streaming.device,
    )

    with patch(
        "escriba.transcribe.streaming.StreamingTranscriber",
        return_value=mock_transcriber,
    ) as mock_cls:
        _build_transcriber(config, realtime_output=True)

    mock_cls.assert_called_once_with(
        model_size=config.streaming.model_size,
        language=config.streaming.language,
        realtime_output=True,
        vad_enabled=config.streaming.vad_enabled,
        vad_config=config.vad,
        hallucination_config=config.hallucination,
        device=config.streaming.device,
    )
