from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_transcriber.transcribe.config import HallucinationConfig, VADConfig
from local_transcriber.utils.env import (
    get_bool_env,
    get_float_env,
    get_int_env,
    get_str_env,
)

logger = logging.getLogger(__name__)


def _load_toml(path: Path) -> dict[str, Any]:
    """
    Load a TOML file into a dict.

    Args:
        path: Path to the TOML file.

    Returns:
        Parsed TOML as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed.
    """
    data = path.read_bytes()
    try:
        import tomllib  # py>=3.11

        return tomllib.loads(data.decode("utf-8"))
    except ModuleNotFoundError:
        # py<=3.10
        import tomli

        return tomli.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse TOML config at {path}") from exc


def resolve_config_path(cli_path: Path | None = None) -> Path | None:
    """
    Resolve the config path to load.

    Precedence:
    - CLI `--config`
    - $LOCAL_TRANSCRIBER_CONFIG
    - ./local-transcriber.toml

    Args:
        cli_path: Config path provided by CLI (optional).

    Returns:
        Resolved Path or None if no config file is found.
    """
    if cli_path is not None:
        return cli_path

    env_path = os.getenv("LOCAL_TRANSCRIBER_CONFIG", "").strip()
    if env_path:
        return Path(env_path)

    default_path = Path("local-transcriber.toml")
    if default_path.exists():
        return default_path

    # Fallback: search upward from cwd for local-transcriber.toml (e.g. run from swift-audio-capture/)
    try:
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / "local-transcriber.toml"
            if candidate.is_file():
                return candidate
    except (IndexError, OSError):
        pass

    # Fallback: package location (editable install: src/local_transcriber/config.py -> project root)
    try:
        _config_file = Path(__file__).resolve()
        project_root = _config_file.parent.parent.parent
        project_toml = project_root / "local-transcriber.toml"
        if project_toml.is_file():
            return project_toml
    except (IndexError, OSError):
        pass

    return None


def _get_section(toml_data: dict[str, Any], section: str) -> dict[str, Any]:
    value = toml_data.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        logger.warning("Invalid TOML section %r (expected table), ignoring", section)
        return {}
    return value


def _toml_has(section: dict[str, Any], key: str) -> bool:
    return key in section and section[key] is not None


def _get_toml_str(section: dict[str, Any], key: str) -> str | None:
    if not _toml_has(section, key):
        return None
    value = section[key]
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    logger.warning("Invalid TOML value for %s (expected string), ignoring", key)
    return None


def _get_toml_int(section: dict[str, Any], key: str) -> int | None:
    if not _toml_has(section, key):
        return None
    value = section[key]
    if isinstance(value, int):
        return value
    logger.warning("Invalid TOML value for %s (expected int), ignoring", key)
    return None


def _get_toml_float(section: dict[str, Any], key: str) -> float | None:
    if not _toml_has(section, key):
        return None
    value = section[key]
    if isinstance(value, (int, float)):
        return float(value)
    logger.warning("Invalid TOML value for %s (expected float), ignoring", key)
    return None


def _get_toml_bool(section: dict[str, Any], key: str) -> bool | None:
    if not _toml_has(section, key):
        return None
    value = section[key]
    if isinstance(value, bool):
        return value
    logger.warning("Invalid TOML value for %s (expected bool), ignoring", key)
    return None


def _get_toml_str_list(section: dict[str, Any], key: str) -> list[str] | None:
    if not _toml_has(section, key):
        return None
    value = section[key]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        cleaned = [v.strip() for v in value if v.strip()]
        return cleaned
    if isinstance(value, str):
        # Allow "txt,json" for convenience.
        cleaned = [v.strip() for v in value.split(",") if v.strip()]
        return cleaned
    logger.warning(
        "Invalid TOML value for %s (expected list[str] or csv string), ignoring", key
    )
    return None


@dataclass(frozen=True)
class SpeakerConfig:
    """
    Speaker configuration.

    Attributes:
        mode: 'none' | 'simple' | 'pyannote'
        threshold: Threshold for simple speaker change detection (0.0-1.0).
    """

    mode: str = "none"
    threshold: float = 0.3


@dataclass(frozen=True)
class StreamingConfig:
    """
    Streaming configuration.
    """

    # NOTE: chunk_duration controls how much audio is buffered before processing.
    # 15s is a good balance between context (accuracy) and latency.
    # Lower values (e.g., 5s) reduce latency; higher values (e.g., 30s) improve accuracy.
    chunk_duration: float = 15.0
    model_size: str = "base"
    language: str = "es"
    device: str = "auto"
    backend: str = "faster-whisper"
    vad_enabled: bool = False
    realtime_output: bool = True
    export_formats: list[str] = field(default_factory=lambda: ["txt", "json"])
    show_metrics: bool = False
    summarize: bool = False
    summary_model: str = "gemini"

    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)


@dataclass(frozen=True)
class AudioConfig:
    """
    Audio capture configuration.
    """

    sample_rate: int = 16000
    channels: int = 1
    mic_only: bool = False
    mic_boost: float = 1.2
    auto_detect_devices: bool = True
    system_device: str = "0"
    mic_device: str = "1"


def _resolve(toml_value, env_fallback_fn):
    """Return toml_value if set, otherwise call env_fallback_fn()."""
    return toml_value if toml_value is not None else env_fallback_fn()


@dataclass(frozen=True)
class AppConfig:
    """
    Full application configuration resolved from TOML + environment.
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)

    config_path: Path | None = None

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
    ) -> AppConfig:
        """
        Load configuration from TOML (if present) and environment.

        TOML takes precedence over environment variables.

        Args:
            config_path: Optional config path. If None, resolves from defaults.

        Returns:
            Resolved AppConfig instance.
        """
        resolved_path = resolve_config_path(config_path)
        toml_data: dict[str, Any] = {}
        if resolved_path is not None:
            try:
                toml_data = _load_toml(resolved_path)
                logger.info("Loaded config from %s", resolved_path)
            except FileNotFoundError:
                logger.warning("Config file not found: %s (continuing)", resolved_path)
            except Exception:
                logger.exception(
                    "Failed to load config file: %s (continuing)", resolved_path
                )

        audio_section = _get_section(toml_data, "audio")
        streaming_section = _get_section(toml_data, "streaming")
        vad_section = _get_section(toml_data, "vad")
        speaker_section = _get_section(toml_data, "speaker")
        whisper_section = _get_section(toml_data, "whisper")

        # Audio
        sample_rate = _get_toml_int(audio_section, "sample_rate")
        channels = _get_toml_int(audio_section, "channels")
        mic_only = _get_toml_bool(audio_section, "mic_only")
        mic_boost = _get_toml_float(audio_section, "mic_boost")
        auto_detect = _get_toml_bool(audio_section, "auto_detect_devices")
        system_device = _get_toml_str(audio_section, "system_device")
        mic_device = _get_toml_str(audio_section, "mic_device")

        audio_cfg = AudioConfig(
            sample_rate=_resolve(sample_rate, lambda: get_int_env("SAMPLE_RATE", 16000, min_value=8000)),
            channels=_resolve(channels, lambda: get_int_env("CHANNELS", 1, min_value=1)),
            mic_only=_resolve(mic_only, lambda: get_bool_env("MIC_ONLY", False)),
            mic_boost=_resolve(mic_boost, lambda: get_float_env("AUDIO_MIC_BOOST", 1.2, min_value=0.1, max_value=5.0)),
            auto_detect_devices=_resolve(auto_detect, lambda: get_bool_env("AUTO_DETECT_DEVICES", True)),
            system_device=_resolve(system_device, lambda: get_str_env("SYSTEM_DEVICE", "0")),
            mic_device=_resolve(mic_device, lambda: get_str_env("MIC_DEVICE", "1")),
        )

        # Streaming
        chunk_duration = _get_toml_float(streaming_section, "chunk_duration")
        model_size = _get_toml_str(streaming_section, "model_size")
        language = _get_toml_str(streaming_section, "language")
        device = _get_toml_str(streaming_section, "device")
        backend = _get_toml_str(streaming_section, "backend")
        vad_enabled = _get_toml_bool(streaming_section, "vad_enabled")
        realtime_output = _get_toml_bool(streaming_section, "realtime_output")
        export_formats = _get_toml_str_list(streaming_section, "export_formats")
        show_metrics = _get_toml_bool(streaming_section, "show_metrics")
        summarize = _get_toml_bool(streaming_section, "summarize")
        summary_model = _get_toml_str(streaming_section, "summary_model")

        # Speaker
        speaker_mode = _get_toml_str(speaker_section, "mode")
        speaker_threshold = _get_toml_float(speaker_section, "threshold")

        speaker_cfg = SpeakerConfig(
            mode=_resolve(speaker_mode, lambda: get_str_env("STREAMING_SPEAKER_MODE", "none")),
            threshold=_resolve(speaker_threshold, lambda: get_float_env("SPEAKER_DETECTION_THRESHOLD", 0.3, min_value=0.0, max_value=1.0)),
        )

        streaming_cfg = StreamingConfig(
            chunk_duration=_resolve(chunk_duration, lambda: get_float_env("STREAMING_CHUNK_DURATION", 15.0, min_value=0.5)),
            model_size=_resolve(model_size, lambda: get_str_env("STREAMING_MODEL_SIZE", "base")),
            language=_resolve(language, lambda: get_str_env("STREAMING_LANGUAGE", "es")),
            device=_resolve(device, lambda: get_str_env("STREAMING_DEVICE", "auto")),
            backend=_resolve(backend, lambda: get_str_env("STREAMING_BACKEND", "faster-whisper")),
            vad_enabled=_resolve(vad_enabled, lambda: get_bool_env("STREAMING_VAD_ENABLED", False)),
            realtime_output=_resolve(realtime_output, lambda: get_bool_env("STREAMING_REALTIME_OUTPUT", True)),
            export_formats=export_formats
            if export_formats is not None
            else (
                [
                    v.strip()
                    for v in get_str_env(
                        "STREAMING_EXPORT_FORMATS", "", allow_empty=True
                    ).split(",")
                    if v.strip()
                ]
                or ["txt", "json"]
            ),
            show_metrics=_resolve(show_metrics, lambda: get_bool_env("STREAMING_SHOW_METRICS", False)),
            summarize=_resolve(summarize, lambda: get_bool_env("STREAMING_SUMMARIZE", False)),
            summary_model=_resolve(summary_model, lambda: get_str_env("STREAMING_SUMMARY_MODEL", "gemini")),
            speaker=speaker_cfg,
        )

        # VAD
        vad_min_silence = _get_toml_int(vad_section, "min_silence_ms")
        vad_threshold = _get_toml_float(vad_section, "threshold")
        vad_cfg = VADConfig(
            min_silence_duration_ms=_resolve(vad_min_silence, lambda: get_int_env("VAD_MIN_SILENCE_MS", 500, min_value=0)),
            threshold=_resolve(vad_threshold, lambda: get_float_env("VAD_THRESHOLD", 0.3, min_value=0.0, max_value=1.0)),
        )

        # Whisper hallucination prevention
        condition_on_previous = _get_toml_bool(
            whisper_section, "condition_on_previous_text"
        )
        no_speech_thresh = _get_toml_float(whisper_section, "no_speech_threshold")
        compression_thresh = _get_toml_float(
            whisper_section, "compression_ratio_threshold"
        )
        logprob_thresh = _get_toml_float(whisper_section, "logprob_threshold")
        hallucination_cfg = HallucinationConfig(
            condition_on_previous_text=_resolve(condition_on_previous, lambda: get_bool_env("WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)),
            no_speech_threshold=_resolve(no_speech_thresh, lambda: get_float_env("WHISPER_NO_SPEECH_THRESHOLD", 0.6, min_value=0.0, max_value=1.0)),
            compression_ratio_threshold=_resolve(compression_thresh, lambda: get_float_env("WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4, min_value=0.0)),
            logprob_threshold=_resolve(logprob_thresh, lambda: get_float_env("WHISPER_LOGPROB_THRESHOLD", -1.0, max_value=0.0)),
        )

        return cls(
            audio=audio_cfg,
            streaming=streaming_cfg,
            vad=vad_cfg,
            hallucination=hallucination_cfg,
            config_path=resolved_path,
        )


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    """
    Convert config to a JSON-serializable dict (for --print-config).
    """
    return {
        "config_path": str(cfg.config_path) if cfg.config_path else None,
        "audio": {
            "sample_rate": cfg.audio.sample_rate,
            "channels": cfg.audio.channels,
            "mic_only": cfg.audio.mic_only,
            "mic_boost": cfg.audio.mic_boost,
            "auto_detect_devices": cfg.audio.auto_detect_devices,
            "system_device": cfg.audio.system_device,
            "mic_device": cfg.audio.mic_device,
        },
        "streaming": {
            "chunk_duration": cfg.streaming.chunk_duration,
            "model_size": cfg.streaming.model_size,
            "language": cfg.streaming.language,
            "device": cfg.streaming.device,
            "backend": cfg.streaming.backend,
            "vad_enabled": cfg.streaming.vad_enabled,
            "realtime_output": cfg.streaming.realtime_output,
            "export_formats": list(cfg.streaming.export_formats),
            "show_metrics": cfg.streaming.show_metrics,
            "summarize": cfg.streaming.summarize,
            "summary_model": cfg.streaming.summary_model,
            "speaker": {
                "mode": cfg.streaming.speaker.mode,
                "threshold": cfg.streaming.speaker.threshold,
            },
        },
        "vad": {
            "min_silence_ms": cfg.vad.min_silence_duration_ms,
            "threshold": cfg.vad.threshold,
        },
        "whisper": {
            "condition_on_previous_text": cfg.hallucination.condition_on_previous_text,
            "no_speech_threshold": cfg.hallucination.no_speech_threshold,
            "compression_ratio_threshold": cfg.hallucination.compression_ratio_threshold,
            "logprob_threshold": cfg.hallucination.logprob_threshold,
        },
    }
