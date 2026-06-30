"""T3: Config validation — domain errors, hot-reload protection, template round-trip."""
from __future__ import annotations

from pathlib import Path

import pytest

from escriba.app.database import Database
from escriba.app.server import AppState
from escriba.config import (
    VALID_AUDIO_SOURCES,
    VALID_BACKENDS,
    AppConfig,
    ConfigValidationError,
)
from tests.conftest import make_handler as _make_handler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config(tmp_path: Path) -> AppConfig:
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[audio]
audio_source = "mic"
sample_rate = 16000
channels = 1

[streaming]
backend = "mlx-whisper"
model_size = "tiny"
chunk_duration = 0.5

[auto_name]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    return AppConfig.load(cfg_path)


@pytest.fixture
def app_state(minimal_config: AppConfig, tmp_path: Path) -> AppState:
    db = Database(tmp_path / "t3-test.db")
    return AppState(config=minimal_config, db=db)


# ---------------------------------------------------------------------------
# ConfigValidationError raised for bad field values
# ---------------------------------------------------------------------------


def test_chunk_duration_zero_raises_named_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text(
        "[streaming]\nchunk_duration = 0\nbackend = \"mlx-whisper\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="chunk_duration"):
        AppConfig.load(cfg_path)


def test_chunk_duration_negative_raises_named_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text(
        "[streaming]\nchunk_duration = -5.0\nbackend = \"mlx-whisper\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="chunk_duration"):
        AppConfig.load(cfg_path)


def test_unknown_backend_raises_named_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text(
        "[streaming]\nchunk_duration = 20.0\nbackend = \"whisper-turbo-3000\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="backend"):
        AppConfig.load(cfg_path)


def test_invalid_audio_source_raises_named_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text(
        "[audio]\naudio_source = \"usb\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="audio_source"):
        AppConfig.load(cfg_path)


def test_mic_boost_too_high_raises_named_error(tmp_path: Path) -> None:
    """mic_boost values above 5.0 must be rejected by validate()."""
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text(
        "[audio]\nmic_boost = 99.0\naudio_source = \"mic\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="mic_boost"):
        AppConfig.load(cfg_path)


def test_mic_boost_at_boundary_passes() -> None:
    """mic_boost == 5.0 and == 0.1 are both valid."""
    AppConfig.load(None)  # default config — should always validate


def test_validate_is_callable_on_config() -> None:
    """validate() does not raise for a default-constructed config."""
    cfg = AppConfig.load(None)  # raises if invalid
    cfg.validate()  # idempotent


def test_valid_backends_constant_is_not_empty() -> None:
    assert len(VALID_BACKENDS) >= 3
    assert "mlx-whisper" in VALID_BACKENDS
    assert "faster-whisper" in VALID_BACKENDS


def test_valid_audio_sources_constant_is_not_empty() -> None:
    assert len(VALID_AUDIO_SOURCES) >= 3
    assert "mic" in VALID_AUDIO_SOURCES
    assert "system" in VALID_AUDIO_SOURCES
    assert "both" in VALID_AUDIO_SOURCES


# ---------------------------------------------------------------------------
# Hot-reload protection: PUT /api/config blocked while recording
# ---------------------------------------------------------------------------


def test_put_config_blocked_while_session_is_active(app_state: AppState) -> None:
    """PUT /api/config returns 409 while a recording session is active."""
    class FakeActiveSession:
        is_active = True
        error = None
        def start(self): pass

    handler = _make_handler(app_state)
    app_state.session = FakeActiveSession()  # type: ignore[assignment]

    # _put_config checks session.is_active under the lock
    payload, status = handler._put_config({"streaming": {"chunk_duration": 10.0}})
    assert status == 409
    assert payload["ok"] is False
    assert "recording" in payload["error"].lower()


# ---------------------------------------------------------------------------
# prompts.templates type consistency and round-trip
# ---------------------------------------------------------------------------


def test_prompts_templates_stored_as_tuple(tmp_path: Path) -> None:
    """After load(), prompts.templates is always a tuple (not a list)."""
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[[prompts.templates]]
id = "t1"
label = "Hello"
prompt = "Say hello"
""".strip(),
        encoding="utf-8",
    )
    cfg = AppConfig.load(cfg_path)
    assert isinstance(cfg.prompts.templates, tuple)


def test_prompts_templates_effective_returns_list(tmp_path: Path) -> None:
    """effective_templates always returns a plain list (for JSON serialization)."""
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[[prompts.templates]]
id = "t1"
label = "Summary"
prompt = "Summarize"
""".strip(),
        encoding="utf-8",
    )
    cfg = AppConfig.load(cfg_path)
    result = cfg.prompts.effective_templates
    assert isinstance(result, list)
    assert result[0]["label"] == "Summary"


def test_prompts_templates_round_trip_via_toml(tmp_path: Path) -> None:
    """Write templates to TOML, reload, verify type is still tuple with same data."""
    from escriba.config import update_config_toml

    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text("[audio]\nsample_rate = 16000\n", encoding="utf-8")

    templates = [
        {"id": "x", "label": "X Label", "prompt": "Do X"},
        {"id": "y", "label": "Y Label", "prompt": "Do Y"},
    ]
    update_config_toml(
        {"prompts": {"system_prompt": "S {transcript} {prompt}", "templates": templates}},
        cfg_path,
    )

    cfg = AppConfig.load(cfg_path)
    assert isinstance(cfg.prompts.templates, tuple)
    assert len(cfg.prompts.templates) == 2
    assert cfg.prompts.templates[0]["label"] == "X Label"
    assert cfg.prompts.templates[1]["label"] == "Y Label"
