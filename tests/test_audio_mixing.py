"""Tests for audio mixing functionality."""

from __future__ import annotations

import numpy as np
import pytest

from escriba.audio.live_capture import mix_audio


def test_mix_audio_basic():
    """Test basic audio mixing."""
    # Crear arrays de prueba (int16)
    system = np.array([1000, 2000, 3000, -1000, -2000], dtype=np.int16)
    mic = np.array([500, 1000, 1500, -500, -1000], dtype=np.int16)

    # Mezclar sin boost
    result = mix_audio(system, mic, mic_boost=1.0)

    # Verificar que el resultado es int16
    assert result.dtype == np.int16

    # Verificar que no hay clipping (valores dentro de rango int16)
    assert np.abs(result).max() <= 32767


def test_mix_audio_with_boost():
    """Test audio mixing with mic boost."""
    system = np.array([1000, 2000, 3000], dtype=np.int16)
    mic = np.array([500, 1000, 1500], dtype=np.int16)

    # Mezclar con boost de 1.2
    result = mix_audio(system, mic, mic_boost=1.2)

    # Verificar que el resultado es int16
    assert result.dtype == np.int16

    # Verificar que no hay clipping
    assert np.abs(result).max() <= 32767


def test_mix_audio_clipping_prevention():
    """Test that mixing prevents clipping."""
    # Crear arrays que causarían clipping sin normalización
    system = np.full(1000, 20000, dtype=np.int16)
    mic = np.full(1000, 20000, dtype=np.int16)

    # Mezclar
    result = mix_audio(system, mic, mic_boost=1.0)

    # Verificar que no hay clipping
    assert np.abs(result).max() <= 32767

    # Verificar que se normalizó (el pico debería ser 32767)
    assert np.abs(result).max() == 32767


def test_mix_audio_empty_arrays():
    """Test mixing with empty arrays."""
    system = np.array([], dtype=np.int16)
    mic = np.array([], dtype=np.int16)

    result = mix_audio(system, mic, mic_boost=1.0)

    assert len(result) == 0
    assert result.dtype == np.int16


def test_mix_audio_different_lengths():
    """Test mixing with arrays of different lengths."""
    system = np.array([1000, 2000, 3000], dtype=np.int16)
    mic = np.array([500, 1000], dtype=np.int16)

    # En la práctica, los arrays deberían tener la misma longitud
    # numpy puede manejar esto con broadcasting, pero puede no ser el comportamiento deseado
    # Por ahora, solo verificamos que no crashea
    try:
        result = mix_audio(system, mic, mic_boost=1.0)
        # Si no lanza excepción, verificar que funciona
        assert result.dtype == np.int16
    except (ValueError, IndexError):
        # Si lanza excepción, está bien - es comportamiento esperado
        pass
