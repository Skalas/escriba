"""Tests for WAV chunk creation."""

from __future__ import annotations

import struct

import pytest

from escriba.audio.live_capture import _create_wav_chunk


def test_create_wav_chunk_basic():
    """Test basic WAV chunk creation."""
    # Crear datos PCM de prueba (int16, mono, 16kHz)
    pcm_data = b"\x00\x00" * 16000  # 1 segundo de silencio

    # Crear header WAV básico
    wav_header = bytearray(44)
    wav_header[0:4] = b"RIFF"
    wav_header[4:8] = struct.pack("<I", 36 + len(pcm_data))
    wav_header[8:12] = b"WAVE"
    wav_header[12:16] = b"fmt "
    wav_header[16:20] = struct.pack("<I", 16)  # fmt chunk size
    wav_header[20:22] = struct.pack("<H", 1)  # audio format (PCM)
    wav_header[22:24] = struct.pack("<H", 1)  # channels (mono)
    wav_header[24:28] = struct.pack("<I", 16000)  # sample rate
    wav_header[28:32] = struct.pack("<I", 32000)  # byte rate
    wav_header[32:34] = struct.pack("<H", 2)  # block align
    wav_header[34:36] = struct.pack("<H", 16)  # bits per sample
    wav_header[36:40] = b"data"
    wav_header[40:44] = struct.pack("<I", len(pcm_data))

    # Crear chunk WAV
    wav_chunk = _create_wav_chunk(
        bytes(wav_header),
        pcm_data,
        n_channels=1,
        sample_rate=16000,
        bits_per_sample=16,
    )

    # Verificar que es un WAV válido
    assert wav_chunk.startswith(b"RIFF")
    assert b"WAVE" in wav_chunk
    assert b"data" in wav_chunk

    # Verificar tamaño mínimo (header + datos)
    assert len(wav_chunk) >= 44 + len(pcm_data)


def test_create_wav_chunk_empty_data():
    """Test WAV chunk creation with empty data."""
    wav_header = bytearray(44)
    wav_header[0:4] = b"RIFF"
    wav_header[8:12] = b"WAVE"

    pcm_data = b""

    wav_chunk = _create_wav_chunk(
        bytes(wav_header),
        pcm_data,
        n_channels=1,
        sample_rate=16000,
        bits_per_sample=16,
    )

    # Debería crear un WAV válido incluso con datos vacíos
    assert wav_chunk.startswith(b"RIFF")
    assert len(wav_chunk) >= 44


def test_create_wav_chunk_stereo():
    """Test WAV chunk creation with stereo audio."""
    # Datos PCM estéreo (2 canales)
    pcm_data = b"\x00\x00" * 32000  # 1 segundo de estéreo

    wav_header = bytearray(44)
    wav_header[0:4] = b"RIFF"
    wav_header[8:12] = b"WAVE"
    wav_header[22:24] = struct.pack("<H", 2)  # channels (stereo)

    wav_chunk = _create_wav_chunk(
        bytes(wav_header),
        pcm_data,
        n_channels=2,
        sample_rate=16000,
        bits_per_sample=16,
    )

    assert wav_chunk.startswith(b"RIFF")
    assert len(wav_chunk) >= 44 + len(pcm_data)
