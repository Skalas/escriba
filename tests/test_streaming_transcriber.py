"""Tests for streaming transcriber functionality."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from escriba.transcribe.config import VADConfig
from escriba.transcribe.metrics import CaptureMetrics
from escriba.transcribe.streaming import StreamingTranscriber


def _create_test_wav(audio_data: bytes, sample_rate: int = 16000) -> bytes:
    """
    Crea un archivo WAV de prueba con los datos de audio dados.

    Args:
        audio_data: Datos PCM en formato int16
        sample_rate: Sample rate en Hz

    Returns:
        Bytes del archivo WAV completo (con header)
    """
    n_channels = 1
    bits_per_sample = 16
    n_samples = len(audio_data) // 2  # 2 bytes por sample (int16)
    data_size = len(audio_data)

    # Crear header WAV
    header = b"RIFF"
    header += struct.pack("<I", 36 + data_size)  # Chunk size
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)  # Subchunk1Size
    header += struct.pack("<H", 1)  # AudioFormat (PCM)
    header += struct.pack("<H", n_channels)  # NumChannels
    header += struct.pack("<I", sample_rate)  # SampleRate
    header += struct.pack("<I", sample_rate * n_channels * bits_per_sample // 8)  # ByteRate
    header += struct.pack("<H", n_channels * bits_per_sample // 8)  # BlockAlign
    header += struct.pack("<H", bits_per_sample)  # BitsPerSample
    header += b"data"
    header += struct.pack("<I", data_size)  # Subchunk2Size

    return header + audio_data


@pytest.fixture
def mock_whisper_model():
    """Fixture para mockear WhisperModel."""
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Hello world"
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_model.transcribe.return_value = ([mock_segment], {"language": "en"})
    return mock_model


def test_streaming_transcriber_init():
    """Test initialization of StreamingTranscriber."""
    with patch("escriba.transcribe.streaming.WhisperModel") as mock_whisper:
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model

        transcriber = StreamingTranscriber(
            model_size="tiny",
            language="en",
            device="cpu",
        )

        assert transcriber.model_size == "tiny"
        assert transcriber.language == "en"
        assert transcriber.device == "cpu"
        mock_whisper.assert_called_once()


def test_streaming_transcriber_with_metrics():
    """Test StreamingTranscriber with metrics enabled."""
    with patch("escriba.transcribe.streaming.WhisperModel") as mock_whisper:
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model

        metrics = CaptureMetrics()
        transcriber = StreamingTranscriber(
            model_size="tiny",
            metrics=metrics,
        )

        assert transcriber.metrics is not None
        assert transcriber.metrics == metrics


def test_process_wav_chunk_with_transcription(mock_whisper_model):
    """Test processing a WAV chunk that produces transcription."""
    with patch("escriba.transcribe.streaming.WhisperModel") as mock_whisper:
        mock_whisper.return_value = mock_whisper_model

        # Crear audio de prueba (1 segundo de silencio)
        sample_rate = 16000
        duration = 1.0
        n_samples = int(sample_rate * duration)
        audio_data = np.zeros(n_samples, dtype=np.int16).tobytes()
        wav_data = _create_test_wav(audio_data, sample_rate)

        transcriber = StreamingTranscriber(model_size="tiny", language="en")

        result = transcriber.process_wav_chunk(wav_data)

        # Verificar que se llamó transcribe
        mock_whisper_model.transcribe.assert_called_once()
        # El resultado debería ser el texto transcrito
        assert result == "Hello world"


def test_process_wav_chunk_silence(mock_whisper_model):
    """Test processing a WAV chunk with silence (no transcription)."""
    with patch("escriba.transcribe.streaming.WhisperModel") as mock_whisper:
        mock_whisper.return_value = mock_whisper_model

        # Configurar mock para retornar segmentos vacíos (silencio)
        mock_whisper_model.transcribe.return_value = ([], {"language": "en"})

        # Crear audio de prueba
        sample_rate = 16000
        duration = 1.0
        n_samples = int(sample_rate * duration)
        audio_data = np.zeros(n_samples, dtype=np.int16).tobytes()
        wav_data = _create_test_wav(audio_data, sample_rate)

        transcriber = StreamingTranscriber(model_size="tiny", language="en")

        result = transcriber.process_wav_chunk(wav_data)

        # Debería retornar None para silencio
        assert result is None


def test_process_wav_chunk_invalid_header():
    """Test processing a WAV chunk with invalid header."""
    with patch("escriba.transcribe.streaming.WhisperModel") as mock_whisper:
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model

        transcriber = StreamingTranscriber(model_size="tiny")

        # Datos inválidos (no es WAV)
        invalid_data = b"INVALID_WAV_DATA"

        result = transcriber.process_wav_chunk(invalid_data)

        # Debería retornar None o manejar el error
        assert result is None


def test_process_wav_chunk_too_small():
    """Test processing a WAV chunk that's too small."""
    with patch("escriba.transcribe.streaming.WhisperModel") as mock_whisper:
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model

        transcriber = StreamingTranscriber(model_size="tiny")

        # Datos muy pequeños (menos de 44 bytes del header)
        small_data = b"RIFF"

        result = transcriber.process_wav_chunk(small_data)

        # Debería retornar None
        assert result is None


def test_process_wav_chunk_with_metrics(mock_whisper_model):
    """Test processing a WAV chunk with metrics tracking."""
    with patch("escriba.transcribe.streaming.WhisperModel") as mock_whisper:
        mock_whisper.return_value = mock_whisper_model

        metrics = CaptureMetrics()
        transcriber = StreamingTranscriber(
            model_size="tiny",
            language="en",
            metrics=metrics,
        )

        # Crear audio de prueba
        sample_rate = 16000
        duration = 1.0
        n_samples = int(sample_rate * duration)
        audio_data = np.zeros(n_samples, dtype=np.int16).tobytes()
        wav_data = _create_test_wav(audio_data, sample_rate)

        initial_chunks = metrics.chunks_processed

        result = transcriber.process_wav_chunk(wav_data)

        # Verificar que se registraron métricas
        assert metrics.chunks_processed > initial_chunks
        assert metrics.chunks_with_transcription > 0 or metrics.chunks_silent > 0


def test_streaming_transcriber_vad_config():
    """Test StreamingTranscriber with custom VAD config."""
    with patch("escriba.transcribe.streaming.WhisperModel") as mock_whisper:
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model

        vad_config = VADConfig(min_silence_duration_ms=1000, threshold=0.5)
        transcriber = StreamingTranscriber(
            model_size="tiny",
            vad_config=vad_config,
        )

        assert transcriber.vad_config.min_silence_duration_ms == 1000
        assert transcriber.vad_config.threshold == 0.5


def test_streaming_transcriber_output_file():
    """Test StreamingTranscriber with output file."""
    with patch("escriba.transcribe.streaming.WhisperModel") as mock_whisper:
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            output_file = Path(tmp.name)

            transcriber = StreamingTranscriber(
                model_size="tiny",
                output_file=output_file,
            )

            assert transcriber.output_file == output_file

            # Limpiar
            output_file.unlink()


def test_process_wav_chunk_error_handling(mock_whisper_model):
    """Test error handling in process_wav_chunk."""
    with patch("escriba.transcribe.streaming.WhisperModel") as mock_whisper:
        mock_whisper.return_value = mock_whisper_model

        # Configurar mock para lanzar excepción
        mock_whisper_model.transcribe.side_effect = Exception("Transcription error")

        transcriber = StreamingTranscriber(model_size="tiny")

        # Crear audio de prueba
        sample_rate = 16000
        duration = 1.0
        n_samples = int(sample_rate * duration)
        audio_data = np.zeros(n_samples, dtype=np.int16).tobytes()
        wav_data = _create_test_wav(audio_data, sample_rate)

        # No debería crashear, debería retornar None
        result = transcriber.process_wav_chunk(wav_data)
        assert result is None
