"""Apple Calendar integration for auto-starting transcriptions."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)

_CALENDAR_PERMISSION_HINT = (
    "Grant Calendar access in System Settings → Privacy & Security → Calendar "
    "for Terminal, iTerm, or Escriba."
)

# Holiday / suggestion calendars make `every event` scans extremely slow.
_SKIP_CALENDAR_NAME_SNIPPETS = (
    "birthday",
    "birthdays",
    "holiday",
    "holidays",
    "festivo",
    "festivos",
    "feriado",
    "siri suggestion",
    "scheduled reminder",
    "días feriados",
    "dias feriados",
)

# Calendar.app event queries routinely exceed 10s with several synced accounts.
_OSASCRIPT_TIMEOUT_SECONDS = 45


def _parse_event_start(time_str: str) -> datetime:
    """Best-effort parse for Apple Calendar / ISO start strings."""
    cleaned = (time_str or "").strip().replace("\u202f", " ").replace("\xa0", " ")
    if not cleaned:
        return datetime.max
    for fmt in (
        "%A, %B %d, %Y at %I:%M:%S %p",
        "%A, %d %B %Y at %I:%M:%S %p",
        "%A, %B %d, %Y at %H:%M:%S",
        "%A, %d %B %Y at %H:%M:%S",
    ):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        return datetime.max


def sort_events_by_start(events: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return events ordered by ascending start time."""
    return sorted(events, key=lambda evt: _parse_event_start(evt.get("start_time", "")))


def _should_skip_calendar(name: str) -> bool:
    lowered = name.strip().lower()
    return any(snippet in lowered for snippet in _SKIP_CALENDAR_NAME_SNIPPETS)


def get_upcoming_events(
    minutes_ahead: int = 5,
    *,
    include_started_within_minutes: int = 120,
) -> tuple[list[dict[str, str]], str | None]:
    """
    Read upcoming events from Apple Calendar via osascript.

    Args:
        minutes_ahead: Minutes into the future to search for events.
        include_started_within_minutes: Also include events that started up to
            this many minutes ago (so an in-progress meeting still appears).

    Returns:
        Tuple of (events, calendar_error). ``calendar_error`` is set when
        Calendar is unavailable or permission was denied; otherwise ``None``.
        Each event dict has ``title``, ``start_time``, ``end_time``, and ``url``.
    """
    try:
        list_result = subprocess.run(
            ["osascript", "-e", 'tell application "Calendar" to get name of every calendar'],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if list_result.returncode != 0:
            stderr = (list_result.stderr or "").strip()
            logger.debug("osascript calendar list error: %s", stderr)
            if "not allowed" in stderr.lower() or "access" in stderr.lower():
                return [], "permission_denied"
            return [], "unavailable"

        calendar_names = [
            name.strip()
            for name in list_result.stdout.split(",")
            if name.strip() and not _should_skip_calendar(name)
        ]
        if not calendar_names:
            return [], None

        # Build per-calendar queries; skipping holidays/birthdays keeps this under timeout.
        cal_blocks: list[str] = []
        for name in calendar_names:
            safe = name.replace("\\", "\\\\").replace('"', '\\"')
            cal_blocks.append(
                f'''
            try
                set cal to calendar "{safe}"
                set eventsList to (every event of cal whose start date ≥ pastDate and start date ≤ futureDate)
                repeat with evt in eventsList
                    set eventUrl to ""
                    try
                        set eventUrl to (url of evt as string)
                    end try
                    set end of outputLines to ((summary of evt as string) & tab & (start date of evt as string) & tab & (end date of evt as string) & tab & eventUrl)
                end repeat
            end try'''
            )

        script = f"""
        tell application "Calendar"
            set nowDate to current date
            set pastDate to nowDate - ({include_started_within_minutes} * minutes)
            set futureDate to nowDate + ({minutes_ahead} * minutes)
            set outputLines to {{}}
            {"".join(cal_blocks)}
            set AppleScript's text item delimiters to linefeed
            return outputLines as text
        end tell
        """

        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=_OSASCRIPT_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            logger.debug("osascript calendar error: %s", stderr)
            if "not allowed" in stderr.lower() or "access" in stderr.lower():
                return [], "permission_denied"
            return [], "unavailable"

        events: list[dict[str, str]] = []
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            title, start_time, end_time = parts[:3]
            url = parts[3] if len(parts) > 3 else ""
            events.append(
                {
                    "title": title,
                    "start_time": start_time,
                    "end_time": end_time,
                    "url": url,
                }
            )
        return sort_events_by_start(events), None

    except subprocess.TimeoutExpired:
        logger.warning("Timeout reading calendar events")
        return [], "unavailable"
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Error reading calendar events: %s", exc, exc_info=True)
        return [], "unavailable"


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
    import threading
    import time

    def watch_loop():
        while True:
            try:
                events, _error = get_upcoming_events(
                    minutes_ahead=notification_minutes + 5
                )
                for event in events:
                    if has_meeting_link(event):
                        callback(event)
                time.sleep(check_interval)
            except Exception as e:
                logger.error("Error in calendar watch loop: %s", e, exc_info=True)
                time.sleep(check_interval)

    thread = threading.Thread(target=watch_loop, daemon=True)
    thread.start()


__all__ = [
    "get_upcoming_events",
    "has_meeting_link",
    "watch_calendar",
    "sort_events_by_start",
    "_CALENDAR_PERMISSION_HINT",
]
