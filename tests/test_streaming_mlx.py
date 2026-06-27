"""Tests for MLX streaming transcriber chunk error handling."""

from __future__ import annotations

import struct
from unittest.mock import patch

import numpy as np
import pytest

from escriba.transcribe.streaming_mlx import (
    CHUNK_INFERENCE_MAX_ATTEMPTS,
    ChunkProcessingError,
    StreamingTranscriberMLX,
)


def _create_test_wav(audio_data: bytes, sample_rate: int = 16000) -> bytes:
    """Build a minimal mono PCM WAV from raw int16 bytes."""
    n_channels = 1
    bits_per_sample = 16
    data_size = len(audio_data)

    header = b"RIFF"
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)
    header += struct.pack("<H", 1)
    header += struct.pack("<H", n_channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", sample_rate * n_channels * bits_per_sample // 8)
    header += struct.pack("<H", n_channels * bits_per_sample // 8)
    header += struct.pack("<H", bits_per_sample)
    header += b"data"
    header += struct.pack("<I", data_size)
    return header + audio_data


@pytest.fixture
def wav_chunk() -> bytes:
    sample_rate = 16000
    duration = 1.0
    n_samples = int(sample_rate * duration)
    audio_data = np.zeros(n_samples, dtype=np.int16).tobytes()
    return _create_test_wav(audio_data, sample_rate)


@pytest.fixture
def mlx_transcriber() -> StreamingTranscriberMLX:
    with patch.object(
        StreamingTranscriberMLX, "_resolve_model_path", return_value="/fake/model"
    ):
        return StreamingTranscriberMLX(model_size="tiny", language="en")


def test_process_wav_chunk_retries_transient_error_then_succeeds(
    mlx_transcriber: StreamingTranscriberMLX, wav_chunk: bytes
) -> None:
    """T2: transient inference failures are retried before succeeding."""
    success_result = {
        "segments": [{"text": " hello ", "start": 0.0, "end": 1.0}],
    }

    with patch("escriba.transcribe.streaming_mlx.mlx_whisper") as mock_mlx, patch(
        "escriba.transcribe.streaming_mlx.time.sleep"
    ):
        mock_mlx.transcribe.side_effect = [
            RuntimeError("Metal hiccup"),
            success_result,
        ]

        result = mlx_transcriber.process_wav_chunk(wav_chunk)

    assert result == "hello"
    assert mock_mlx.transcribe.call_count == 2


def test_process_wav_chunk_raises_after_exhausted_retries(
    mlx_transcriber: StreamingTranscriberMLX, wav_chunk: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    """T2: exhausted retries surface ChunkProcessingError instead of silent None."""
    caplog.set_level("ERROR")

    with patch("escriba.transcribe.streaming_mlx.mlx_whisper") as mock_mlx, patch(
        "escriba.transcribe.streaming_mlx.time.sleep"
    ):
        mock_mlx.transcribe.side_effect = RuntimeError("Metal hiccup")

        with pytest.raises(ChunkProcessingError):
            mlx_transcriber.process_wav_chunk(wav_chunk)

    assert mock_mlx.transcribe.call_count == CHUNK_INFERENCE_MAX_ATTEMPTS
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_process_wav_chunk_invalid_wav_still_returns_none(
    mlx_transcriber: StreamingTranscriberMLX,
) -> None:
    """Invalid WAV input remains a silent no-op, not a chunk processing error."""
    result = mlx_transcriber.process_wav_chunk(b"NOT_A_WAV")
    assert result is None
