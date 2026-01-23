"""
Detección automática de llamadas activas en macOS.
Similar a Notion AI, detecta cuando estás en una llamada y puede iniciar transcripción automáticamente.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Procesos comunes de videollamadas
MEETING_APPS = {
    "zoom": ["zoom", "ZoomOpener", "zTray"],
    "teams": ["Microsoft Teams", "teams"],
    "meet": ["Google Chrome", "Google Meet"],
    "webex": ["Cisco Webex", "webexmta"],
    "skype": ["Skype", "Skype for Business"],
    "facetime": ["FaceTime"],
    "discord": ["Discord"],
    "slack": ["Slack"],
}


def is_meeting_app_running() -> bool:
    """
    Detecta si alguna aplicación de videollamadas está corriendo.

    Returns:
        True si hay una app de meeting activa
    """
    try:
        # Listar procesos activos
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=2,
        )

        processes = result.stdout.lower()

        # Buscar procesos de meeting apps
        for app_name, process_names in MEETING_APPS.items():
            for process_name in process_names:
                if process_name.lower() in processes:
                    logger.debug(
                        f"Found meeting app: {app_name} (process: {process_name})"
                    )
                    return True

        return False
    except Exception as e:
        logger.error(f"Error checking meeting apps: {e}", exc_info=True)
        return False


def is_audio_active() -> bool:
    """
    Detecta si hay actividad de audio en el sistema.

    Usa un método simple: verifica si hay procesos usando audio.
    En una implementación más avanzada, se podría usar Core Audio
    para detectar streams de audio activos.

    Returns:
        True si hay actividad de audio
    """
    try:
        # Verificar si hay procesos usando dispositivos de audio
        # Esto es una aproximación - una implementación más precisa
        # requeriría usar Core Audio APIs
        result = subprocess.run(
            ["lsof", "-n", "-P", "-i", "TCP"],
            capture_output=True,
            text=True,
            timeout=2,
        )

        # Si hay conexiones de red activas y apps de meeting corriendo,
        # probablemente hay una llamada
        if is_meeting_app_running():
            # Verificar si hay conexiones de red activas (indicador de llamada)
            output = result.stdout.lower()
            # Buscar patrones comunes de videollamadas
            video_call_indicators = [
                "zoom.us",
                "teams.microsoft.com",
                "meet.google.com",
                "webex.com",
            ]

            for indicator in video_call_indicators:
                if indicator in output:
                    logger.debug(f"Found audio activity indicator: {indicator}")
                    return True

        return False
    except Exception as e:
        logger.debug(f"Error checking audio activity: {e}")
        # Fallback: si hay app de meeting, asumir que hay audio
        return is_meeting_app_running()


def detect_active_call() -> tuple[bool, Optional[str]]:
    """
    Detecta si hay una llamada activa.

    Returns:
        Tuple (is_call_active, app_name)
        app_name será None si no se puede determinar
    """
    # Verificar apps de meeting
    for app_name, process_names in MEETING_APPS.items():
        try:
            result = subprocess.run(
                ["pgrep", "-f", process_names[0]],
                capture_output=True,
                timeout=1,
            )
            if result.returncode == 0:
                logger.info(f"Detected active call in: {app_name}")
                return True, app_name
        except Exception:
            continue

    return False, None


def wait_for_call_start(
    check_interval: float = 2.0,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """
    Espera hasta que se detecte una llamada activa.

    Args:
        check_interval: Intervalo en segundos entre verificaciones
        timeout: Tiempo máximo de espera (None = infinito)

    Returns:
        Nombre de la app de meeting detectada o None si timeout
    """
    start_time = time.time()
    logger.info("Waiting for call to start...")

    while True:
        is_active, app_name = detect_active_call()

        if is_active:
            logger.info(f"Call detected in: {app_name}")
            return app_name

        if timeout and (time.time() - start_time) > timeout:
            logger.debug("Timeout waiting for call")
            return None

        time.sleep(check_interval)


def monitor_call_status(
    callback_on_start,
    callback_on_end,
    check_interval: float = 2.0,
) -> None:
    """
    Monitorea el estado de llamadas y ejecuta callbacks.

    Args:
        callback_on_start: Función a llamar cuando inicia una llamada
        callback_on_end: Función a llamar cuando termina una llamada
        check_interval: Intervalo en segundos entre verificaciones
    """
    was_in_call = False

    while True:
        is_active, app_name = detect_active_call()

        if is_active and not was_in_call:
            # Llamada inició
            logger.info(f"Call started in: {app_name}")
            callback_on_start(app_name)
            was_in_call = True
        elif not is_active and was_in_call:
            # Llamada terminó
            logger.info("Call ended")
            callback_on_end()
            was_in_call = False

        time.sleep(check_interval)
