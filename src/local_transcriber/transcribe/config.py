from __future__ import annotations

from dataclasses import dataclass

from local_transcriber.utils.env import get_float_env, get_int_env


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
