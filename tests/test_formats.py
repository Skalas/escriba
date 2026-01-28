"""Tests for export formats."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from local_transcriber.transcribe.formats import (
    export_to_json,
    export_to_markdown,
    export_to_srt,
    export_to_txt,
)


@pytest.fixture
def sample_segments():
    """Sample segments for testing."""
    return [
        {"start": 0.0, "end": 3.5, "text": "Hola, ¿cómo están todos?"},
        {
            "start": 3.5,
            "end": 7.2,
            "text": "Perfecto, yo me encargo de la documentación.",
        },
        {
            "start": 7.2,
            "end": 10.0,
            "text": "Excelente, entonces nos vemos la próxima semana.",
        },
    ]


def test_export_to_txt(sample_segments):
    """Test TXT export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.txt"
        export_to_txt(sample_segments, output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")

        # Verificar que contiene el texto
        assert "Hola, ¿cómo están todos?" in content
        assert "[00:00:00.000]" in content


def test_export_to_json(sample_segments):
    """Test JSON export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.json"
        metadata = {"model": "base", "language": "es"}
        export_to_json(sample_segments, output_path, metadata)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")

        # Verificar estructura JSON
        assert '"segments"' in content
        assert '"metadata"' in content
        assert "Hola, ¿cómo están todos?" in content


def test_export_to_srt(sample_segments):
    """Test SRT export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.srt"
        export_to_srt(sample_segments, output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")

        # Verificar formato SRT
        assert "00:00:00,000 --> 00:00:03,500" in content
        assert "Hola, ¿cómo están todos?" in content
        assert "1\n" in content  # Primera entrada


def test_export_to_markdown(sample_segments):
    """Test Markdown export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.md"
        export_to_markdown(sample_segments, output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")

        # Verificar formato Markdown
        assert "# Transcripción" in content
        assert "## [00:00:00]" in content
        assert "Hola, ¿cómo están todos?" in content


def test_export_empty_segments():
    """Test export with empty segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.txt"
        export_to_txt([], output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert content == ""


def test_export_segments_with_empty_text():
    """Test export with segments that have empty text."""
    segments = [
        {"start": 0.0, "end": 3.5, "text": "Hola"},
        {"start": 3.5, "end": 7.2, "text": ""},  # Empty text
        {"start": 7.2, "end": 10.0, "text": "Adiós"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.txt"
        export_to_txt(segments, output_path)

        content = output_path.read_text(encoding="utf-8")
        # El segmento vacío no debería aparecer o aparecer sin texto
        assert "Hola" in content
        assert "Adiós" in content
