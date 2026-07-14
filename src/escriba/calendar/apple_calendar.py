"""Apple Calendar integration for Up next and settings."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

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

# One calendar at a time is usually <1s; large synced accounts can take ~15s.
_PER_CALENDAR_TIMEOUT_SECONDS = 20
_LIST_CALENDARS_TIMEOUT_SECONDS = 15
_CACHE_TTL_SECONDS = 45.0
_MAX_CALENDAR_WORKERS = 4

_cache_lock = threading.Lock()
_cache_key: tuple[Any, ...] | None = None
_cache_at: float = 0.0
_cache_value: tuple[list[dict[str, str]], str | None] = ([], None)


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


def _parse_event_lines(stdout: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for line in stdout.splitlines():
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
    return events


def _list_calendar_names() -> tuple[list[str], str | None]:
    """List calendar names from Calendar.app.

    Returns:
        Tuple of (names, calendar_error). ``calendar_error`` is set when
        Calendar is unavailable or permission was denied; otherwise ``None``.
    """
    try:
        list_result = subprocess.run(
            [
                "osascript",
                "-e",
                (
                    'tell application "Calendar"\n'
                    "    set AppleScript's text item delimiters to linefeed\n"
                    "    return (name of every calendar) as text\n"
                    "end tell"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=_LIST_CALENDARS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Timeout listing calendars")
        return [], "unavailable"
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Error listing calendars: %s", exc, exc_info=True)
        return [], "unavailable"

    if list_result.returncode != 0:
        stderr = (list_result.stderr or "").strip()
        logger.debug("osascript calendar list error: %s", stderr)
        if "not allowed" in stderr.lower() or "access" in stderr.lower():
            return [], "permission_denied"
        return [], "unavailable"

    names = _parse_calendar_names_stdout(list_result.stdout)
    return names, None


def _parse_calendar_names_stdout(stdout: str) -> list[str]:
    """Parse calendar names from osascript list output (one name per line)."""
    names: list[str] = []
    for line in stdout.splitlines():
        name = line.strip()
        if name and is_safe_calendar_name(name):
            names.append(name)
    return names


def is_safe_calendar_name(name: str) -> bool:
    """Return False for names that cannot be safely embedded in osascript."""
    cleaned = name.strip()
    if not cleaned:
        return False
    if "\\" in cleaned:
        return False
    for ch in cleaned:
        code = ord(ch)
        if code < 32 or ch in ('"', "\n", "\r"):
            return False
    return True


def _allowlist_set(
    calendar_allowlist: tuple[str, ...] | list[str] | None,
) -> set[str]:
    """Normalize configured calendar names into a safe allowlist set."""
    return {
        name.strip()
        for name in (calendar_allowlist or ())
        if is_safe_calendar_name(name)
    }


def _filter_calendar_names(
    names: list[str],
    calendar_allowlist: list[str] | None = None,
) -> list[str]:
    """Apply holiday skip list and optional user allowlist.

    An empty allowlist means all non-skipped calendars (backward compatible).
    """
    candidates = [
        name
        for name in names
        if is_safe_calendar_name(name) and not _should_skip_calendar(name)
    ]
    allow_set = _allowlist_set(calendar_allowlist)
    if not allow_set:
        return candidates
    return [name for name in candidates if name in allow_set]


def describe_calendars_for_settings(
    calendar_allowlist: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[dict[str, str | bool]], str | None]:
    """Describe calendars for the Settings multi-select.

    Args:
        calendar_allowlist: Configured allowlist. Empty means all non-skipped
            calendars are treated as selected.

    Returns:
        Tuple of (entries, calendar_error). Each entry has ``name``, ``skipped``,
        and ``selected`` keys.
    """
    names, error = _list_calendar_names()
    if error:
        return [], error

    allow = _allowlist_set(calendar_allowlist)
    use_all = not allow
    entries: list[dict[str, str | bool]] = []
    for name in sorted(names, key=str.lower):
        if not is_safe_calendar_name(name):
            continue
        skipped = _should_skip_calendar(name)
        if skipped:
            selected = False
        elif use_all:
            selected = True
        else:
            selected = name in allow
        entries.append({"name": name, "skipped": skipped, "selected": selected})
    return entries, None


def _query_calendar_events(
    calendar_name: str,
    *,
    minutes_ahead: int,
    include_started_within_minutes: int,
) -> list[dict[str, str]]:
    """Fetch events for a single Calendar.app calendar."""
    if not is_safe_calendar_name(calendar_name):
        logger.warning("Skipping unsafe calendar name: %r", calendar_name)
        return []
    safe = calendar_name.replace("\\", "\\\\").replace('"', '\\"')
    script = f"""
    tell application "Calendar"
        set nowDate to current date
        set pastDate to nowDate - ({include_started_within_minutes} * minutes)
        set futureDate to nowDate + ({minutes_ahead} * minutes)
        set cal to calendar "{safe}"
        set eventsList to (every event of cal whose start date ≥ pastDate and start date ≤ futureDate)
        set outputLines to {{}}
        repeat with evt in eventsList
            set eventUrl to ""
            try
                set eventUrl to (url of evt as string)
            end try
            set end of outputLines to ((summary of evt as string) & tab & (start date of evt as string) & tab & (end date of evt as string) & tab & eventUrl)
        end repeat
        set AppleScript's text item delimiters to linefeed
        return outputLines as text
    end tell
    """
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=_PER_CALENDAR_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        logger.debug("osascript calendar %r error: %s", calendar_name, stderr)
        if "not allowed" in stderr.lower() or "access" in stderr.lower():
            raise PermissionError(stderr or "Calendar permission denied")
        return []
    return _parse_event_lines(result.stdout)


def _fetch_upcoming_events(
    minutes_ahead: int,
    *,
    include_started_within_minutes: int,
    calendar_allowlist: list[str] | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    """Query Calendar.app without caching (caller holds the lock)."""
    all_names, list_error = _list_calendar_names()
    if list_error:
        return [], list_error

    calendar_names = _filter_calendar_names(all_names, calendar_allowlist)
    if not calendar_names:
        if calendar_allowlist and _filter_calendar_names(all_names, None):
            return [], "no_matching_calendars"
        return [], None

    events: list[dict[str, str]] = []
    timed_out = 0
    permission_denied = False

    # Prefer personal/named calendars before slow Exchange/email accounts.
    calendar_names.sort(key=lambda n: (("@" in n), n.lower()))

    workers = min(_MAX_CALENDAR_WORKERS, len(calendar_names))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _query_calendar_events,
                name,
                minutes_ahead=minutes_ahead,
                include_started_within_minutes=include_started_within_minutes,
            ): name
            for name in calendar_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                events.extend(future.result())
            except PermissionError:
                permission_denied = True
            except subprocess.TimeoutExpired:
                timed_out += 1
                logger.warning("Timeout reading calendar %r", name)
            except (OSError, subprocess.SubprocessError) as exc:
                logger.error(
                    "Error reading calendar %r: %s", name, exc, exc_info=True
                )

    if events:
        return sort_events_by_start(events), None
    if permission_denied:
        return [], "permission_denied"
    if timed_out:
        return [], "unavailable"
    return [], None


def get_upcoming_events(
    minutes_ahead: int = 5,
    *,
    include_started_within_minutes: int = 120,
    calendar_allowlist: list[str] | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    """
    Read upcoming events from Apple Calendar via osascript.

    Queries calendars individually (in parallel) and coalesces concurrent
    callers behind a short in-process cache so Home double-fetches do not
    stack competing Calendar.app scripts.

    Args:
        minutes_ahead: Minutes into the future to search for events.
        include_started_within_minutes: Also include events that started up to
            this many minutes ago (so an in-progress meeting still appears).
        calendar_allowlist: Optional list of calendar names to query. Empty or
            ``None`` queries all non-skipped calendars.

    Returns:
        Tuple of (events, calendar_error). ``calendar_error`` is set when
        Calendar is unavailable or permission was denied; otherwise ``None``.
        Each event dict has ``title``, ``start_time``, ``end_time``, and ``url``.
    """
    global _cache_key, _cache_at, _cache_value
    allow_key = tuple(sorted(name.strip() for name in (calendar_allowlist or []) if name.strip()))
    key = (minutes_ahead, include_started_within_minutes, allow_key)
    with _cache_lock:
        now = time.monotonic()
        if _cache_key == key and (now - _cache_at) < _CACHE_TTL_SECONDS:
            return _cache_value

        events, error = _fetch_upcoming_events(
            minutes_ahead,
            include_started_within_minutes=include_started_within_minutes,
            calendar_allowlist=calendar_allowlist,
        )
        _cache_key = key
        _cache_at = time.monotonic()
        _cache_value = (events, error)
        return events, error


def clear_upcoming_events_cache() -> None:
    """Drop the in-process upcoming-events cache (tests / forced refresh)."""
    global _cache_key, _cache_at, _cache_value
    with _cache_lock:
        _cache_key = None
        _cache_at = 0.0
        _cache_value = ([], None)


__all__ = [
    "get_upcoming_events",
    "clear_upcoming_events_cache",
    "describe_calendars_for_settings",
    "is_safe_calendar_name",
    "sort_events_by_start",
    "_CALENDAR_PERMISSION_HINT",
]
