"""Apple Calendar integration for auto-starting transcriptions."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def get_upcoming_events(minutes_ahead: int = 5) -> list[dict[str, str]]:
    """
    Obtiene eventos próximos de Apple Calendar.

    Args:
        minutes_ahead: Minutos hacia adelante para buscar eventos

    Returns:
        Lista de eventos con 'title', 'start_time', 'end_time', 'url'
    """
    try:
        # Usar icalBuddy o EventKit
        # Por simplicidad, usamos osascript para leer Calendar
        script = f"""
        tell application "Calendar"
            set nowDate to current date
            set futureDate to nowDate + ({minutes_ahead} * minutes)
            
            set upcomingEvents to {{}}
            repeat with cal in calendars
                set eventsList to (every event of cal whose start date is greater than nowDate and start date is less than futureDate)
                repeat with evt in eventsList
                    set eventInfo to {{}}
                    set end of eventInfo to (summary of evt as string)
                    set end of eventInfo to (start date of evt as string)
                    set end of eventInfo to (end date of evt as string)
                    set end of eventInfo to (url of evt as string)
                    set end of upcomingEvents to eventInfo
                end repeat
            end repeat
            return upcomingEvents
        end tell
        """
        
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.debug("osascript error: %s", result.stderr)
            return []

        # Parsear resultado (formato puede variar)
        events = []
        # Por ahora retornar lista vacía - implementación completa requeriría
        # parsing más robusto del output de osascript
        return events

    except subprocess.TimeoutExpired:
        logger.warning("Timeout reading calendar events")
        return []
    except Exception as e:
        logger.error("Error reading calendar events: %s", e, exc_info=True)
        return []


def has_meeting_link(event: dict[str, str]) -> bool:
    """
    Verifica si un evento tiene un link de reunión.

    Args:
        event: Diccionario con información del evento

    Returns:
        True si tiene link de Zoom/Meet/Teams
    """
    url = event.get("url", "").lower()
    title = event.get("title", "").lower()
    
    meeting_keywords = ["zoom", "meet", "teams", "webex", "gotomeeting"]
    
    return any(keyword in url or keyword in title for keyword in meeting_keywords)


def watch_calendar(
    callback,
    check_interval: int = 60,
    notification_minutes: int = 1,
) -> None:
    """
    Observa el calendario y llama al callback cuando hay eventos próximos.

    Args:
        callback: Función a llamar con información del evento
        check_interval: Intervalo en segundos para verificar calendario
        notification_minutes: Minutos antes del evento para notificar
    """
    import time
    import threading

    def watch_loop():
        while True:
            try:
                events = get_upcoming_events(minutes_ahead=notification_minutes + 5)
                for event in events:
                    if has_meeting_link(event):
                        callback(event)
                time.sleep(check_interval)
            except Exception as e:
                logger.error("Error in calendar watch loop: %s", e, exc_info=True)
                time.sleep(check_interval)

    thread = threading.Thread(target=watch_loop, daemon=True)
    thread.start()
