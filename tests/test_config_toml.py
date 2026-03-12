from __future__ import annotations

from pathlib import Path

import pytest

from escriba.config import AppConfig


def test_toml_overrides_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    TOML should override environment variables when both are set.
    """
    monkeypatch.setenv("SAMPLE_RATE", "8000")
    monkeypatch.setenv("STREAMING_MODEL_SIZE", "tiny")

    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[audio]
sample_rate = 16000

[streaming]
model_size = "base"
""".strip(),
        encoding="utf-8",
    )

    cfg = AppConfig.load(cfg_path)
    assert cfg.audio.sample_rate == 16000
    assert cfg.streaming.model_size == "base"


def test_env_used_when_toml_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SAMPLE_RATE", "8000")
    missing_path = tmp_path / "does-not-exist.toml"
    monkeypatch.setenv("ESCRIBA_CONFIG", str(missing_path))
    cfg = AppConfig.load(None)
    assert cfg.audio.sample_rate == 8000
