"""Tests for calendar CLI and read-only integration."""
from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from escriba.cli import app


def test_watch_calendar_auto_start_blocked() -> None:
    """--auto-start remains blocked with a clear error (T15)."""
    runner = CliRunner()
    result = runner.invoke(app, ["watch-calendar", "--auto-start"])
    assert result.exit_code != 0
    assert "auto-start is not implemented" in result.output.lower()


def test_get_upcoming_events_permission_denied() -> None:
    from escriba.calendar.apple_calendar import get_upcoming_events

    fake = type(
        "R",
        (),
        {"returncode": 1, "stdout": "", "stderr": "Not allowed to send Apple events"},
    )()
    with patch("escriba.calendar.apple_calendar.subprocess.run", return_value=fake):
        events, error = get_upcoming_events(minutes_ahead=30)
    assert events == []
    assert error == "permission_denied"


def test_sort_events_by_start_orders_earliest_first() -> None:
    from escriba.calendar.apple_calendar import sort_events_by_start

    events = [
        {
            "title": "Later",
            "start_time": "2026-07-13T15:00:00",
            "end_time": "",
            "url": "",
        },
        {
            "title": "Sooner",
            "start_time": "2026-07-13T09:00:00",
            "end_time": "",
            "url": "",
        },
    ]
    ordered = sort_events_by_start(events)
    assert [e["title"] for e in ordered] == ["Sooner", "Later"]


def test_parse_event_start_accepts_day_month_locale() -> None:
    from escriba.calendar.apple_calendar import _parse_event_start

    parsed = _parse_event_start("Monday, 13 July 2026 at 10:30:00\u202fPM")
    assert parsed.year == 2026
    assert parsed.month == 7
    assert parsed.day == 13
    assert parsed.hour == 22
    assert parsed.minute == 30


def test_should_skip_holiday_calendars() -> None:
    from escriba.calendar.apple_calendar import _should_skip_calendar

    assert _should_skip_calendar("Birthdays")
    assert _should_skip_calendar("Holidays in Mexico")
    assert not _should_skip_calendar("Work")
