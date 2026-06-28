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


def test_prompts_default_when_absent(tmp_path: Path) -> None:
    """Without a [prompts] section, effective values fall back to defaults."""
    from escriba.config import DEFAULT_PROMPT_TEMPLATES, DEFAULT_SYSTEM_PROMPT

    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text("[audio]\nsample_rate = 16000\n", encoding="utf-8")

    cfg = AppConfig.load(cfg_path)
    assert cfg.prompts.effective_system_prompt == DEFAULT_SYSTEM_PROMPT
    assert len(cfg.prompts.effective_templates) == len(DEFAULT_PROMPT_TEMPLATES)


def test_prompts_loaded_from_toml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[prompts]
system_prompt = "Do {prompt} on {transcript}"

[[prompts.templates]]
id = "tldr"
label = "TL;DR"
prompt = "Summarize in one line"
""".strip(),
        encoding="utf-8",
    )

    cfg = AppConfig.load(cfg_path)
    assert cfg.prompts.effective_system_prompt == "Do {prompt} on {transcript}"
    templates = cfg.prompts.effective_templates
    assert len(templates) == 1
    assert templates[0]["label"] == "TL;DR"


def test_prompts_skip_incomplete_templates(tmp_path: Path) -> None:
    """Templates missing label or prompt are dropped on load."""
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[[prompts.templates]]
label = "Valid"
prompt = "do it"

[[prompts.templates]]
label = "No prompt"
""".strip(),
        encoding="utf-8",
    )

    cfg = AppConfig.load(cfg_path)
    assert [t["label"] for t in cfg.prompts.templates] == ["Valid"]


def test_update_config_toml_preserves_sections(tmp_path: Path) -> None:
    """Merging prompts must not drop existing sections."""
    from escriba.config import _load_toml, update_config_toml

    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        '[audio]\nmic_boost = 1.4\n\n[streaming]\nmodel_size = "medium"\n',
        encoding="utf-8",
    )

    update_config_toml(
        {"prompts": {"system_prompt": "X {transcript} {prompt}", "templates": []}},
        cfg_path,
    )

    data = _load_toml(cfg_path)
    assert data["audio"]["mic_boost"] == 1.4
    assert data["streaming"]["model_size"] == "medium"
    assert data["prompts"]["system_prompt"] == "X {transcript} {prompt}"


def test_auto_record_round_trip_from_toml(tmp_path: Path) -> None:
    """All auto_record keys load from TOML and serialize back."""
    from escriba.config import config_to_dict

    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[auto_record]
enabled = true
start_mode = "auto"
cooldown_seconds = 45
poll_interval = 2
start_debounce_seconds = 2.5
stop_debounce_seconds = 4.0
""".strip(),
        encoding="utf-8",
    )

    cfg = AppConfig.load(cfg_path)
    assert cfg.auto_record.enabled is True
    assert cfg.auto_record.start_mode == "auto"
    assert cfg.auto_record.cooldown_seconds == 45
    assert cfg.auto_record.poll_interval == 2
    assert cfg.auto_record.start_debounce_seconds == 2.5
    assert cfg.auto_record.stop_debounce_seconds == 4.0

    serialized = config_to_dict(cfg)["auto_record"]
    assert serialized == {
        "enabled": True,
        "start_mode": "auto",
        "cooldown_seconds": 45,
        "poll_interval": 2,
        "start_debounce_seconds": 2.5,
        "stop_debounce_seconds": 4.0,
    }


def test_auto_record_invalid_start_mode_falls_back_to_prompt(tmp_path: Path) -> None:
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        '[auto_record]\nstart_mode = "instant"\n',
        encoding="utf-8",
    )

    cfg = AppConfig.load(cfg_path)
    assert cfg.auto_record.start_mode == "prompt"


def test_t6_put_auto_record_partial_save_deep_merges(tmp_path: Path) -> None:
    """T6: PUT [auto_record] partial save deep-merges; other sections intact."""
    from escriba.config import _load_toml, update_config_toml

    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[audio]
mic_boost = 1.4

[streaming]
model_size = "medium"

[auto_record]
enabled = false
cooldown_seconds = 60
""".strip(),
        encoding="utf-8",
    )

    update_config_toml(
        {
            "auto_record": {
                "enabled": True,
                "start_mode": "prompt",
                "start_debounce_seconds": 3.0,
                "stop_debounce_seconds": 5.0,
            }
        },
        cfg_path,
    )

    data = _load_toml(cfg_path)
    assert data["audio"]["mic_boost"] == 1.4
    assert data["streaming"]["model_size"] == "medium"
    assert data["auto_record"]["enabled"] is True
    assert data["auto_record"]["start_mode"] == "prompt"
    assert data["auto_record"]["cooldown_seconds"] == 60
    assert data["auto_record"]["start_debounce_seconds"] == 3.0
    assert data["auto_record"]["stop_debounce_seconds"] == 5.0
