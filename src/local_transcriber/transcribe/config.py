from __future__ import annotations

from dataclasses import dataclass

from local_transcriber.utils.env import get_bool_env, get_float_env, get_int_env


@dataclass
class HallucinationConfig:
    """
    Configuration for Whisper hallucination prevention.

    These parameters help prevent repetitive output (e.g., "los los los...")
    that can occur during silence or low-quality audio.

    Attributes:
        condition_on_previous_text: If False, prevents repetition loops by not
            using previous transcription as context. Default: False.
        no_speech_threshold: Threshold for filtering silent segments.
            Higher values filter more aggressively. Default: 0.6.
        compression_ratio_threshold: Threshold for detecting repetitive patterns.
            Segments with compression ratio above this are filtered. Default: 2.4.
        logprob_threshold: Threshold for filtering low-confidence segments.
            Segments with average log probability below this are filtered. Default: -1.0.
    """

    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0

    @classmethod
    def from_env(cls) -> HallucinationConfig:
        """
        Create HallucinationConfig from environment variables.

        Environment variables:
            WHISPER_CONDITION_ON_PREVIOUS_TEXT: Use previous text as context (default: false)
            WHISPER_NO_SPEECH_THRESHOLD: No-speech filter threshold (default: 0.6)
            WHISPER_COMPRESSION_RATIO_THRESHOLD: Repetition filter threshold (default: 2.4)
            WHISPER_LOGPROB_THRESHOLD: Low-confidence filter threshold (default: -1.0)

        Returns:
            HallucinationConfig instance with values from environment or defaults
        """
        return cls(
            condition_on_previous_text=get_bool_env(
                "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False
            ),
            no_speech_threshold=get_float_env(
                "WHISPER_NO_SPEECH_THRESHOLD", 0.6, min_value=0.0, max_value=1.0
            ),
            compression_ratio_threshold=get_float_env(
                "WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4, min_value=0.0
            ),
            logprob_threshold=get_float_env(
                "WHISPER_LOGPROB_THRESHOLD", -1.0, max_value=0.0
            ),
        )


@dataclass
class VADConfig:
    """
    Configuración para Voice Activity Detection (VAD).

    Attributes:
        min_silence_duration_ms: Duración mínima de silencio en milisegundos
            antes de considerar que terminó un segmento de voz.
        threshold: Umbral de sensibilidad (0.0-1.0). Valores más bajos
            detectan más voz pero pueden incluir ruido.
    """

    min_silence_duration_ms: int = 500
    threshold: float = 0.3

    @classmethod
    def from_env(cls) -> VADConfig:
        """
        Crea una instancia de VADConfig desde variables de entorno.

        Variables de entorno:
            VAD_MIN_SILENCE_MS: Duración mínima de silencio en ms (default: 500)
            VAD_THRESHOLD: Umbral de sensibilidad (default: 0.3)

        Returns:
            Instancia de VADConfig con valores de entorno o defaults
        """
        min_silence_ms = get_int_env("VAD_MIN_SILENCE_MS", 500, min_value=0)
        threshold = get_float_env("VAD_THRESHOLD", 0.3, min_value=0.0, max_value=1.0)

        return cls(
            min_silence_duration_ms=min_silence_ms,
            threshold=threshold,
        )
