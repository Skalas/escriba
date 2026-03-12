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


def _is_meeting_app_running() -> bool:
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
                        "Found meeting app: %s (process: %s)", app_name, process_name
                    )
                    return True

        return False
    except Exception as e:
        logger.error("Error checking meeting apps: %s", e, exc_info=True)
        return False


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
                logger.info("Detected active call in: %s", app_name)
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
            logger.info("Call detected in: %s", app_name)
            return app_name

        if timeout and (time.time() - start_time) > timeout:
            logger.debug("Timeout waiting for call")
            return None

        time.sleep(check_interval)


