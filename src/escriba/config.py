from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from escriba.transcribe.config import HallucinationConfig, VADConfig
from escriba.utils.env import (
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
    - $ESCRIBA_CONFIG (with fallback to deprecated $LOCAL_TRANSCRIBER_CONFIG)
    - ./escriba.toml (with fallback to deprecated ./local-transcriber.toml)

    Args:
        cli_path: Config path provided by CLI (optional).

    Returns:
        Resolved Path or None if no config file is found.
    """
    if cli_path is not None:
        return cli_path

    env_path = os.getenv("ESCRIBA_CONFIG", "").strip()
    if not env_path:
        # Backward compat: check deprecated env var
        env_path = os.getenv("LOCAL_TRANSCRIBER_CONFIG", "").strip()
        if env_path:
            logger.warning(
                "LOCAL_TRANSCRIBER_CONFIG is deprecated, use ESCRIBA_CONFIG instead"
            )
    if env_path:
        return Path(env_path)

    default_path = Path("escriba.toml")
    if default_path.exists():
        return default_path

    # Backward compat: check deprecated config filename
    legacy_path = Path("local-transcriber.toml")
    if legacy_path.exists():
        logger.warning(
            "local-transcriber.toml is deprecated, rename to escriba.toml"
        )
        return legacy_path

    # Fallback: search upward from cwd for escriba.toml (e.g. run from swift-audio-capture/)
    try:
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / "escriba.toml"
            if candidate.is_file():
                return candidate
            # Backward compat
            legacy_candidate = parent / "local-transcriber.toml"
            if legacy_candidate.is_file():
                logger.warning(
                    "local-transcriber.toml is deprecated, rename to escriba.toml"
                )
                return legacy_candidate
    except (IndexError, OSError):
        pass

    # Fallback: package location (editable install: src/escriba/config.py -> project root)
    try:
        _config_file = Path(__file__).resolve()
        project_root = _config_file.parent.parent.parent
        project_toml = project_root / "escriba.toml"
        if project_toml.is_file():
            return project_toml
        # Backward compat
        legacy_toml = project_root / "local-transcriber.toml"
        if legacy_toml.is_file():
            logger.warning(
                "local-transcriber.toml is deprecated, rename to escriba.toml"
            )
            return legacy_toml
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
class DictionaryConfig:
    """Custom vocabulary for improving transcription accuracy.

    Attributes:
        terms: Domain terms/acronyms fed to Whisper's initial_prompt to bias recognition.
        replacements: Post-processing find-and-replace map (case-insensitive keys).
    """

    terms: list[str] = field(default_factory=list)
    replacements: dict[str, str] = field(default_factory=dict)

    @property
    def initial_prompt(self) -> str | None:
        if not self.terms:
            return None
        return ", ".join(self.terms)

    def apply_replacements(self, text: str) -> str:
        if not self.replacements:
            return text
        for wrong, correct in self.replacements.items():
            # Case-insensitive replace preserving boundaries
            import re
            text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
        return text


@dataclass(frozen=True)
class SpeakerConfig:
    """
    Speaker configuration.

    Attributes:
        mode: 'none' | 'simple' | 'pyannote'
        threshold: Threshold for simple speaker change detection (0.0-1.0).
    """

    mode: str = "pyannote"
    threshold: float = 0.3


@dataclass(frozen=True)
class StreamingConfig:
    """
    Streaming configuration.
    """

    # NOTE: chunk_duration controls how much audio is buffered before processing.
    # 15s is a good balance between context (accuracy) and latency.
    # Lower values (e.g., 5s) reduce latency; higher values (e.g., 30s) improve accuracy.
    chunk_duration: float = 20.0
    model_size: str = "medium"
    language: str = "auto"
    device: str = "auto"
    backend: str = "mlx-whisper"
    vad_enabled: bool = True
    realtime_output: bool = True
    export_formats: list[str] = field(default_factory=lambda: ["txt", "json"])
    show_metrics: bool = False
    summarize: bool = False
    summary_model: str = "auto"

    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)


@dataclass(frozen=True)
class AutoRecordConfig:
    """Mic-activation detection: prompt user to record when mic is active."""

    enabled: bool = False
    cooldown_seconds: int = 60
    poll_interval: int = 3


@dataclass(frozen=True)
class AutoNameConfig:
    """Automatic session naming via LLM."""

    enabled: bool = True
    min_segments: int = 5
    max_snippet_words: int = 500


@dataclass(frozen=True)
class LocalLLMConfig:
    """Local LLM configuration for on-device inference via mlx-lm."""

    model: str = "auto"  # "auto" = pick by RAM, or explicit HF repo ID
    cache_ttl: int = 300  # seconds to keep model loaded after last use


# Default AI-notes system prompt. {transcript} and {prompt} are required
# placeholders: the transcript text and the user's per-notes instruction.
# XML tags delimit the inputs so the model never confuses transcript content
# with instructions.
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert meeting and call notetaker. You turn raw transcripts "
    "into accurate, clearly structured notes.\n\n"
    "<transcript>\n"
    "{transcript}\n"
    "</transcript>\n\n"
    "<task>\n"
    "{prompt}\n"
    "</task>\n\n"
    "<instructions>\n"
    "- Base every statement strictly on the transcript. Never invent names, "
    "numbers, decisions, or events that are not present.\n"
    "- If the transcript lacks the information the task needs, say so plainly "
    "instead of guessing.\n"
    "- Write in the SAME LANGUAGE as the transcript. If it is in Spanish, use "
    "natural, fluent Spanish with correct orthography: acentos (á, é, í, ó, ú), "
    "ñ, ü, and the opening signs ¿ and ¡.\n"
    "- Format with clean Markdown (headings, bullet or numbered lists, bold for "
    "key terms). Do not wrap the whole reply in a code block.\n"
    "- Be concise and skimmable. Omit filler and meta-commentary about the task.\n"
    "</instructions>"
)

# Default quick-prompt templates shown as chips in the dashboard.
DEFAULT_PROMPT_TEMPLATES: list[dict[str, str]] = [
    {"id": "summary", "label": "Executive Summary", "prompt": "Write an executive summary in 3-5 sentences covering the purpose of the conversation, the main topics, the key conclusions, and any agreed next steps. Use clear, direct prose — no lists."},
    {"id": "actions", "label": "Action Items", "prompt": "Extract every action item, task, and commitment as a Markdown checklist. For each, include the action, the owner (if stated), and the deadline or timeframe (if mentioned). Group by owner when possible. If no action items were discussed, state that clearly."},
    {"id": "decisions", "label": "Decisions", "prompt": "List every decision that was made. For each, give: (1) what was decided, (2) the rationale or context, and (3) any alternatives discussed and rejected. If no decisions were reached, say so."},
    {"id": "questions", "label": "Open Questions", "prompt": "List all unresolved questions, open items, and topics that need follow-up. For each, note who raised it (if mentioned) and what information or action is needed to resolve it. If nothing is open, state that."},
    {"id": "keypoints", "label": "Key Points", "prompt": "Summarize the most important points, insights, and takeaways as a bulleted list. Capture specific figures, names, and data mentioned. Focus on substance, not a chronological recap."},
    {"id": "blockers", "label": "Risks & Blockers", "prompt": "Identify all risks, blockers, dependencies, and concerns raised. For each, note the issue, who flagged it, and any proposed mitigation or workaround. If none were raised, state that."},
    {"id": "minutes", "label": "Meeting Minutes", "prompt": "Write structured meeting minutes in Markdown with these sections: **Attendees**, **Agenda**, **Discussion**, **Decisions**, **Action Items**, **Next Steps**. Omit a section only if there is genuinely nothing to record under it."},
    {"id": "interview", "label": "Interview Evaluation", "prompt": "Critically evaluate the candidate in this interview transcript. Do NOT be charitable or complacent: most candidates do not meet the bar, and your job is to find the evidence, not to give the benefit of the doubt. For each relevant competency (technical depth, problem-solving, communication, ownership and impact, and any role-specific skills discussed), give a rating (Strong / Adequate / Weak / No evidence) and cite the specific transcript evidence behind it. Sharply distinguish what the candidate DEMONSTRATED from what they merely CLAIMED. Flag vague, generic, evasive, or buzzword answers, contradictions, and moments where they dodged a question or could not go deep when pressed. Structure the output as: (1) Recommendation — one of Strong Hire / Hire / No Hire / Strong No Hire, with a one-line justification and your confidence; (2) Competency assessment — rating + evidence per competency; (3) Strengths — concrete, evidenced positives (omit if none); (4) Concerns and red flags — weaknesses, gaps, evasions, contradictions; (5) Missing signal — what the interview did not probe or the candidate did not substantiate, and what evidence would be needed to raise the rating. Hold a high bar: if evidence for a competency is thin or absent, rate it Weak or No evidence rather than assuming competence. When the overall signal is mixed or insufficient, lean toward No Hire and state exactly what evidence is missing."},
    {"id": "followup", "label": "Follow-up Email", "prompt": "Draft a concise, professional follow-up email: a one-line recap, the decisions made, action items with owners, and next steps. Open with a greeting and close with a clear call to action."},
]


@dataclass(frozen=True)
class PromptsConfig:
    """User-customizable AI prompts: the system prompt and quick templates."""

    system_prompt: str = ""  # empty => fall back to DEFAULT_SYSTEM_PROMPT
    templates: tuple[dict[str, str], ...] = ()  # empty => DEFAULT_PROMPT_TEMPLATES

    @property
    def effective_system_prompt(self) -> str:
        return self.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT

    @property
    def effective_templates(self) -> list[dict[str, str]]:
        return [dict(t) for t in self.templates] if self.templates else list(DEFAULT_PROMPT_TEMPLATES)


@dataclass(frozen=True)
class AudioConfig:
    """
    Audio capture configuration.
    """

    sample_rate: int = 16000
    channels: int = 1
    mic_only: bool = False
    audio_source: str = "both"  # "system" | "mic" | "both"
    mic_boost: float = 1.4
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
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    auto_record: AutoRecordConfig = field(default_factory=AutoRecordConfig)
    auto_name: AutoNameConfig = field(default_factory=AutoNameConfig)
    local_llm: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)

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
        audio_source = _get_toml_str(audio_section, "audio_source")
        mic_boost = _get_toml_float(audio_section, "mic_boost")
        auto_detect = _get_toml_bool(audio_section, "auto_detect_devices")
        system_device = _get_toml_str(audio_section, "system_device")
        mic_device = _get_toml_str(audio_section, "mic_device")

        # Resolve audio_source: prefer explicit setting, fall back to mic_only compat
        resolved_mic_only = _resolve(mic_only, lambda: get_bool_env("MIC_ONLY", False))
        if audio_source is None:
            resolved_audio_source = "mic" if resolved_mic_only else "both"
        else:
            resolved_audio_source = audio_source

        audio_cfg = AudioConfig(
            sample_rate=_resolve(sample_rate, lambda: get_int_env("SAMPLE_RATE", 16000, min_value=8000)),
            channels=_resolve(channels, lambda: get_int_env("CHANNELS", 1, min_value=1)),
            mic_only=resolved_mic_only,
            audio_source=resolved_audio_source,
            mic_boost=_resolve(mic_boost, lambda: get_float_env("AUDIO_MIC_BOOST", 1.4, min_value=0.1, max_value=5.0)),
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
            mode=_resolve(speaker_mode, lambda: get_str_env("STREAMING_SPEAKER_MODE", "pyannote")),
            threshold=_resolve(speaker_threshold, lambda: get_float_env("SPEAKER_DETECTION_THRESHOLD", 0.3, min_value=0.0, max_value=1.0)),
        )

        streaming_cfg = StreamingConfig(
            chunk_duration=_resolve(chunk_duration, lambda: get_float_env("STREAMING_CHUNK_DURATION", 20.0, min_value=0.5)),
            model_size=_resolve(model_size, lambda: get_str_env("STREAMING_MODEL_SIZE", "medium")),
            language=_resolve(language, lambda: get_str_env("STREAMING_LANGUAGE", "auto")),
            device=_resolve(device, lambda: get_str_env("STREAMING_DEVICE", "auto")),
            backend=_resolve(backend, lambda: get_str_env("STREAMING_BACKEND", "mlx-whisper")),
            vad_enabled=_resolve(vad_enabled, lambda: get_bool_env("STREAMING_VAD_ENABLED", True)),
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

        # Auto-record (mic activation detection)
        ar_section = _get_section(toml_data, "auto_record")
        ar_enabled = _get_toml_bool(ar_section, "enabled")
        ar_cooldown = _get_toml_int(ar_section, "cooldown_seconds")
        ar_poll = _get_toml_int(ar_section, "poll_interval")
        auto_record_cfg = AutoRecordConfig(
            enabled=ar_enabled if ar_enabled is not None else False,
            cooldown_seconds=ar_cooldown if ar_cooldown is not None else 60,
            poll_interval=ar_poll if ar_poll is not None else 3,
        )

        # Auto-name (LLM session naming)
        an_section = _get_section(toml_data, "auto_name")
        an_enabled = _get_toml_bool(an_section, "enabled")
        an_min_seg = _get_toml_int(an_section, "min_segments")
        an_max_words = _get_toml_int(an_section, "max_snippet_words")
        auto_name_cfg = AutoNameConfig(
            enabled=an_enabled if an_enabled is not None else True,
            min_segments=an_min_seg if an_min_seg is not None else 5,
            max_snippet_words=an_max_words if an_max_words is not None else 500,
        )

        # Local LLM
        ll_section = _get_section(toml_data, "local_llm")
        ll_model = _get_toml_str(ll_section, "model")
        ll_ttl = _get_toml_int(ll_section, "cache_ttl")
        local_llm_cfg = LocalLLMConfig(
            model=ll_model if ll_model is not None else "auto",
            cache_ttl=ll_ttl if ll_ttl is not None else 300,
        )

        # Dictionary (custom vocabulary)
        dict_section = _get_section(toml_data, "dictionary")
        dict_terms = _get_toml_str_list(dict_section, "terms") or []
        dict_replacements_raw = dict_section.get("replacements", {})
        dict_replacements = {}
        if isinstance(dict_replacements_raw, dict):
            dict_replacements = {
                str(k): str(v) for k, v in dict_replacements_raw.items()
            }
        dict_cfg = DictionaryConfig(terms=dict_terms, replacements=dict_replacements)

        # Prompts (custom AI system prompt + quick templates)
        prompts_section = _get_section(toml_data, "prompts")
        p_system = _get_toml_str(prompts_section, "system_prompt")
        p_templates_raw = prompts_section.get("templates", [])
        p_templates: list[dict[str, str]] = []
        if isinstance(p_templates_raw, list):
            for item in p_templates_raw:
                if isinstance(item, dict) and item.get("label") and item.get("prompt"):
                    p_templates.append(
                        {
                            "id": str(item.get("id") or item["label"]),
                            "label": str(item["label"]),
                            "prompt": str(item["prompt"]),
                        }
                    )
        prompts_cfg = PromptsConfig(
            system_prompt=p_system or "",
            templates=tuple(p_templates),
        )

        return cls(
            audio=audio_cfg,
            streaming=streaming_cfg,
            vad=vad_cfg,
            hallucination=hallucination_cfg,
            dictionary=dict_cfg,
            auto_record=auto_record_cfg,
            auto_name=auto_name_cfg,
            local_llm=local_llm_cfg,
            prompts=prompts_cfg,
            config_path=resolved_path,
        )


def save_config_to_toml(config_dict: dict[str, Any], path: Path) -> None:
    """Write a config dict to a TOML file."""
    import tomli_w

    # Remove non-TOML keys
    writable = {k: v for k, v in config_dict.items() if k != "config_path"}
    path.write_bytes(tomli_w.dumps(writable).encode("utf-8"))
    logger.info("Saved config to %s", path)


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` (returns a new dict)."""
    result = dict(base)
    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def update_config_toml(updates: dict[str, Any], path: Path) -> None:
    """
    Merge ``updates`` into the existing TOML file and write it back.

    Unlike :func:`save_config_to_toml`, this preserves sections that are not
    present in ``updates`` instead of overwriting the whole file.
    """
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = _load_toml(path)
        except Exception:
            logger.warning("Could not parse existing config at %s; rewriting", path)
    merged = _deep_merge(existing, {k: v for k, v in updates.items() if k != "config_path"})
    save_config_to_toml(merged, path)


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
            "audio_source": cfg.audio.audio_source,
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
        "dictionary": {
            "terms": list(cfg.dictionary.terms),
            "replacements": dict(cfg.dictionary.replacements),
        },
        "auto_record": {
            "enabled": cfg.auto_record.enabled,
            "cooldown_seconds": cfg.auto_record.cooldown_seconds,
            "poll_interval": cfg.auto_record.poll_interval,
        },
        "auto_name": {
            "enabled": cfg.auto_name.enabled,
            "min_segments": cfg.auto_name.min_segments,
            "max_snippet_words": cfg.auto_name.max_snippet_words,
        },
        "local_llm": {
            "model": cfg.local_llm.model,
            "cache_ttl": cfg.local_llm.cache_ttl,
        },
        "prompts": {
            "system_prompt": cfg.prompts.effective_system_prompt,
            "templates": cfg.prompts.effective_templates,
            "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
            "default_templates": list(DEFAULT_PROMPT_TEMPLATES),
        },
    }
