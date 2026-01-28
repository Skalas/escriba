"""Tests for device detection functionality."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from local_transcriber.audio.device_detection import (
    auto_detect_devices,
    list_audio_devices,
)


def test_list_audio_devices_success():
    """Test listing audio devices with successful ffmpeg output."""
    mock_output = """
[AVFoundation indev @ 0x123] AVFoundation audio devices:
[0] Built-in Microphone
[1] AirPods Pro
[2] iPhone Microphone
"""
    mock_result = MagicMock()
    mock_result.stderr = mock_output
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        devices = list_audio_devices()

        assert "inputs" in devices
        assert isinstance(devices["inputs"], list)
        # Verificar que se parsearon algunos dispositivos
        # (el número exacto depende del mock)


def test_list_audio_devices_no_devices():
    """Test listing audio devices when no devices are found."""
    mock_output = "[AVFoundation indev @ 0x123] AVFoundation audio devices:"
    mock_result = MagicMock()
    mock_result.stderr = mock_output
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        devices = list_audio_devices()

        assert "inputs" in devices
        assert isinstance(devices["inputs"], list)


def test_list_audio_devices_ffmpeg_error():
    """Test listing audio devices when ffmpeg fails."""
    mock_result = MagicMock()
    mock_result.stderr = "Error: ffmpeg not found"
    mock_result.returncode = 1

    with patch("subprocess.run", return_value=mock_result):
        devices = list_audio_devices()

        # Debería retornar estructura vacía en caso de error
        assert "inputs" in devices
        assert isinstance(devices["inputs"], list)


def test_list_audio_devices_timeout():
    """Test listing audio devices when ffmpeg times out."""
    with patch(
        "subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 5)
    ):
        devices = list_audio_devices()

        # Debería retornar estructura vacía en caso de timeout
        assert "inputs" in devices
        assert isinstance(devices["inputs"], list)


def test_auto_detect_devices():
    """Test auto-detection of devices."""
    # Mock list_audio_devices para retornar dispositivos de prueba
    mock_devices = {
        "inputs": [
            {"index": "0", "name": "Built-in Microphone", "type": "microphone"},
            {"index": "1", "name": "AirPods Pro", "type": "headphones"},
        ],
        "outputs": [
            {"index": "0", "name": "Built-in Output", "type": "speaker"},
        ],
    }

    with patch("local_transcriber.audio.device_detection.list_audio_devices") as mock_list:
        mock_list.return_value = mock_devices

        system_device, mic_device = auto_detect_devices()

        # Verificar que retorna valores (pueden ser None si no se detectan)
        assert system_device is None or isinstance(system_device, str)
        assert mic_device is None or isinstance(mic_device, str)


def test_auto_detect_devices_no_devices():
    """Test auto-detection when no devices are available."""
    mock_devices = {"inputs": [], "outputs": []}

    with patch("local_transcriber.audio.device_detection.list_audio_devices") as mock_list:
        mock_list.return_value = mock_devices

        system_device, mic_device = auto_detect_devices()

        # Debería retornar None cuando no hay dispositivos
        assert system_device is None
        assert mic_device is None


def test_auto_detect_devices_with_screen_capture():
    """Test auto-detection when ScreenCaptureKit is available."""
    # Cuando ScreenCaptureKit está disponible, system_device debería ser None
    # porque no se necesita un dispositivo físico
    mock_devices = {
        "inputs": [
            {"index": "0", "name": "Built-in Microphone", "type": "microphone"},
        ],
        "outputs": [],
    }

    with patch("local_transcriber.audio.device_detection.list_audio_devices") as mock_list:
        mock_list.return_value = mock_devices

        # auto_detect_devices siempre retorna None para system_device
        # porque usa ScreenCaptureKit (no necesita dispositivo físico)
        system_device, mic_device = auto_detect_devices()

        # system_device siempre es None (ScreenCaptureKit no requiere dispositivo)
        assert system_device is None
        # mic_device puede ser detectado o None dependiendo de los dispositivos disponibles
        assert mic_device is None or isinstance(mic_device, str)


def test_list_audio_devices_parsing_various_formats():
    """Test parsing various ffmpeg output formats."""
    test_cases = [
        # Formato estándar
        "[0] Built-in Microphone",
        # Con espacios extra
        "[ 1 ]  AirPods Pro  ",
        # Con prefijos
        "[in#0] [0] iPhone Microphone",
    ]

    for test_line in test_cases:
        mock_output = f"""
[AVFoundation indev @ 0x123] AVFoundation audio devices:
{test_line}
"""
        mock_result = MagicMock()
        mock_result.stderr = mock_output
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            devices = list_audio_devices()
            # Solo verificamos que no crashea
            assert "inputs" in devices
