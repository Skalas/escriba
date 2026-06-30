"""Verify that deprecated/unavailable backends fall back to a real StreamingTranscriber."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from escriba.config import AppConfig


def _config(tmp_path: Path, backend: str) -> AppConfig:
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        f"""
[audio]
audio_source = "mic"
sample_rate = 16000
channels = 1

[streaming]
backend = "{backend}"
model_size = "tiny"
chunk_duration = 0.5

[auto_name]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    return AppConfig.load(cfg_path)


class _AbortAfterConstruction(Exception):
    """Raised by the fake transcriber to abort run_streaming_capture early."""


@pytest.mark.parametrize("backend", ["openai-whisper", "mlx-whisper", "faster-whisper"])
def test_all_backends_construct_a_streaming_transcriber(
    tmp_path: Path, backend: str
) -> None:
    """Every backend path must result in a constructed StreamingTranscriber."""
    config = _config(tmp_path, backend)
    constructed: list[str] = []

    class _FakeTranscriber:
        def __init__(self, **kwargs):
            constructed.append(backend)
            raise _AbortAfterConstruction

    with (
        patch("escriba.audio.live_capture.StreamingTranscriber", _FakeTranscriber),
        patch("escriba.audio.live_capture._load_mlx_transcriber", return_value=None),
    ):
        with pytest.raises(_AbortAfterConstruction):
            from escriba.audio.live_capture import run_streaming_capture

            run_streaming_capture(tmp_path, config=config)

    assert constructed, (
        f"StreamingTranscriber was never constructed for backend={backend!r}; "
        "fallback is broken"
    )


def test_mlx_available_uses_mlx_and_skips_faster_whisper(tmp_path: Path) -> None:
    """When mlx-whisper IS available, the mlx class is used and StreamingTranscriber is not."""
    config = _config(tmp_path, "mlx-whisper")
    mlx_constructed: list[bool] = []
    fw_constructed: list[bool] = []

    class _FakeMlxTranscriber:
        def __init__(self, **kwargs):
            mlx_constructed.append(True)
            raise _AbortAfterConstruction

    class _FakeFasterWhisper:
        def __init__(self, **kwargs):
            fw_constructed.append(True)
            raise _AbortAfterConstruction

    with (
        patch("escriba.audio.live_capture.StreamingTranscriber", _FakeFasterWhisper),
        patch("escriba.audio.live_capture._load_mlx_transcriber", return_value=_FakeMlxTranscriber),
    ):
        with pytest.raises(_AbortAfterConstruction):
            from escriba.audio.live_capture import run_streaming_capture

            run_streaming_capture(tmp_path, config=config)

    assert mlx_constructed, "mlx transcriber was never constructed on the happy path"
    assert not fw_constructed, (
        "StreamingTranscriber (faster-whisper) was constructed when mlx was available; "
        "the `if transcriber is None` guard is wrongly shadowing the mlx path"
    )
