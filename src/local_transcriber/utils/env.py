from __future__ import annotations

import os


def get_str_env(name: str, default: str, allow_empty: bool = False) -> str:
    """
    Obtiene una variable de entorno como string.

    Args:
        name: Nombre de la variable de entorno
        default: Valor por defecto si no existe
        allow_empty: Si True, permite valores vacíos (default: False)

    Returns:
        Valor de la variable de entorno o default

    Raises:
        ValueError: Si la variable está vacía y allow_empty es False
    """
    value = os.getenv(name, default).strip()
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return value


def get_int_env(name: str, default: int, min_value: int | None = None) -> int:
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


def get_float_env(
    name: str,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
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


def get_bool_env(name: str, default: bool) -> bool:
    """
    Obtiene una variable de entorno como bool.

    Args:
        name: Nombre de la variable de entorno
        default: Valor por defecto si no existe

    Returns:
        True si el valor es "true", "1", "yes", o "on" (case-insensitive)
    """
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in ("true", "1", "yes", "on")
