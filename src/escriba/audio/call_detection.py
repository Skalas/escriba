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


def _find_running_meeting_app() -> Optional[str]:
    """
    Best-effort lookup of a known meeting app process.

    Returns:
        App label (e.g. ``zoom``) or None if no known app is running.
    """
    for app_name, process_names in MEETING_APPS.items():
        try:
            result = subprocess.run(
                ["pgrep", "-f", process_names[0]],
                capture_output=True,
                timeout=1,
            )
            if result.returncode == 0:
                logger.debug("Found meeting app process: %s", app_name)
                return app_name
        except Exception:
            continue
    return None


def detect_active_call() -> tuple[bool, Optional[str]]:
    """
    Detecta si hay una llamada activa (CLI / legacy path).

    Uses process presence only — suitable for ``wait_for_call_start`` but not
    for menubar auto-record, which must gate on mic activity.

    Returns:
        Tuple (is_call_active, app_name)
        app_name será None si no se puede determinar
    """
    app_name = _find_running_meeting_app()
    if app_name is not None:
        logger.info("Detected active call in: %s", app_name)
        return True, app_name
    return False, None


def get_call_app_label_if_mic_active(mic_running: bool) -> Optional[str]:
    """
    Return a meeting-app label only when the mic is active and a known app matches.

    Background process presence alone is not treated as an active call.
    """
    if not mic_running:
        return None
    return _find_running_meeting_app()


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


