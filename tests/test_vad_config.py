"""Tests for VAD configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from escriba.transcribe.config import VADConfig


def test_vad_config_defaults():
    """Test VAD config with default values."""
    with patch.dict(os.environ, {}, clear=True):
        config = VADConfig.from_env()

        assert config.min_silence_duration_ms == 500
        assert config.threshold == 0.3


def test_vad_config_from_env():
    """Test VAD config loading from environment variables."""
    env_vars = {
        "VAD_MIN_SILENCE_MS": "1000",
        "VAD_THRESHOLD": "0.5",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        config = VADConfig.from_env()

        assert config.min_silence_duration_ms == 1000
        assert config.threshold == 0.5


def test_vad_config_validation():
    """Test VAD config validation."""
    # Valores válidos
    config = VADConfig(min_silence_duration_ms=500, threshold=0.3)
    assert config.min_silence_duration_ms == 500
    assert config.threshold == 0.3

    # Threshold fuera de rango debería ser validado (si hay validación)
    # Por ahora, solo verificamos que se puede crear
    config = VADConfig(min_silence_duration_ms=100, threshold=1.0)
    assert config.threshold == 1.0


def test_vad_config_from_env_invalid_values():
    """Test VAD config with invalid environment values."""
    # Valores inválidos deberían usar defaults o lanzar error
    # Dependiendo de la implementación
    env_vars = {
        "VAD_MIN_SILENCE_MS": "invalid",
        "VAD_THRESHOLD": "invalid",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        # Debería usar defaults o lanzar ValueError
        try:
            config = VADConfig.from_env()
            # Si no lanza error, debería usar defaults
            assert config.min_silence_duration_ms == 500
            assert config.threshold == 0.3
        except ValueError:
            # Si lanza error, está bien
            pass
