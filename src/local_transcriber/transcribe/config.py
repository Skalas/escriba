from __future__ import annotations

import os
from dataclasses import dataclass


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
        min_silence_ms = _get_int_env("VAD_MIN_SILENCE_MS", 500, min_value=0)
        threshold = _get_float_env("VAD_THRESHOLD", 0.3, min_value=0.0, max_value=1.0)

        return cls(
            min_silence_duration_ms=min_silence_ms,
            threshold=threshold,
        )


def _get_int_env(name: str, default: int, min_value: int | None = None) -> int:
    """
    Obtiene una variable de entorno como entero.
    
    Args:
        name: Nombre de la variable de entorno
        default: Valor por defecto si no existe
        min_value: Valor mínimo permitido (opcional)
    
    Returns:
        Valor de la variable de entorno como int
    
    Raises:
        ValueError: Si el valor no es un entero válido o es menor que min_value
    """
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _get_float_env(
    name: str, default: float, min_value: float | None = None, max_value: float | None = None
) -> float:
    """
    Obtiene una variable de entorno como float.
    
    Args:
        name: Nombre de la variable de entorno
        default: Valor por defecto si no existe
        min_value: Valor mínimo permitido (opcional)
        max_value: Valor máximo permitido (opcional)
    
    Returns:
        Valor de la variable de entorno como float
    
    Raises:
        ValueError: Si el valor no es un float válido o está fuera de rango
    """
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return value
