"""Tests for batch file transcription errors."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from escriba.transcribe.whisper import TranscriptionError, transcribe_file


def test_transcribe_file_returns_none_for_missing_input(tmp_path: Path) -> None:
    result = transcribe_file(tmp_path / "missing.wav", tmp_path / "out")
    assert result is None


def test_transcribe_file_raises_on_subprocess_failure(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    output = tmp_path / "out"

    with patch(
        "escriba.transcribe.whisper.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "whisper"),
    ):
        with pytest.raises(TranscriptionError, match="Whisper failed"):
            transcribe_file(audio, output)


def test_transcribe_file_raises_when_transcript_missing(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    output = tmp_path / "out"

    with patch("escriba.transcribe.whisper.subprocess.run"):
        with pytest.raises(TranscriptionError, match="Transcript not found"):
            transcribe_file(audio, output)
