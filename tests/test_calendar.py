"""Tests for calendar integration."""
from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from escriba.cli import app


def test_watch_calendar_command_removed() -> None:
    """Orphan watch-calendar CLI removed; calendar auto-start stays parked (T7/T8)."""
    runner = CliRunner()
    result = runner.invoke(app, ["watch-calendar"])
    assert result.exit_code != 0


def test_get_upcoming_events_permission_denied() -> None:
    from escriba.calendar import apple_calendar
    from escriba.calendar.apple_calendar import (
        clear_upcoming_events_cache,
        get_upcoming_events,
    )

    clear_upcoming_events_cache()
    fake = type(
        "R",
        (),
        {"returncode": 1, "stdout": "", "stderr": "Not allowed to send Apple events"},
    )()
    with patch.object(apple_calendar.subprocess, "run", return_value=fake):
        events, error = get_upcoming_events(minutes_ahead=30)
    assert events == []
    assert error == "permission_denied"


def test_get_upcoming_events_cache_coalesces_repeat_calls() -> None:
    from escriba.calendar import apple_calendar
    from escriba.calendar.apple_calendar import (
        clear_upcoming_events_cache,
        get_upcoming_events,
    )

    clear_upcoming_events_cache()
    list_result = type(
        "R",
        (),
        {"returncode": 0, "stdout": "Work\nBirthdays\n", "stderr": ""},
    )()
    event_result = type(
        "R",
        (),
        {
            "returncode": 0,
            "stdout": "Standup\t2026-07-13T10:00:00\t2026-07-13T10:30:00\t\n",
            "stderr": "",
        },
    )()
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        script = " ".join(cmd)
        calls.append(script)
        if "name of every calendar" in script:
            return list_result
        return event_result

    with patch.object(apple_calendar.subprocess, "run", side_effect=fake_run):
        first, err1 = get_upcoming_events(minutes_ahead=30)
        second, err2 = get_upcoming_events(minutes_ahead=30)

    assert err1 is None and err2 is None
    assert [e["title"] for e in first] == ["Standup"]
    assert second == first
    # List once + one Work query; Birthdays skipped; second call is cache hit.
    assert sum("name of every calendar" in c for c in calls) == 1
    assert sum('calendar "Work"' in c for c in calls) == 1


def test_get_upcoming_events_empty_allowlist_queries_all_non_skipped() -> None:
    from escriba.calendar import apple_calendar
    from escriba.calendar.apple_calendar import (
        clear_upcoming_events_cache,
        get_upcoming_events,
    )

    clear_upcoming_events_cache()
    list_result = type(
        "R",
        (),
        {"returncode": 0, "stdout": "Work\nBirthdays\nPersonal\n", "stderr": ""},
    )()
    event_result = type(
        "R",
        (),
        {"returncode": 0, "stdout": "", "stderr": ""},
    )()
    queried: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        script = " ".join(cmd)
        if "name of every calendar" in script:
            return list_result
        for name in ("Work", "Personal"):
            if f'calendar "{name}"' in script:
                queried.append(name)
        return event_result

    with patch.object(apple_calendar.subprocess, "run", side_effect=fake_run):
        events, error = get_upcoming_events(minutes_ahead=30, calendar_allowlist=[])

    assert error is None
    assert events == []
    assert sorted(queried) == ["Personal", "Work"]


def test_get_upcoming_events_allowlist_filters_calendars() -> None:
    from escriba.calendar import apple_calendar
    from escriba.calendar.apple_calendar import (
        clear_upcoming_events_cache,
        get_upcoming_events,
    )

    clear_upcoming_events_cache()
    list_result = type(
        "R",
        (),
        {"returncode": 0, "stdout": "Work\nPersonal\nExchange\n", "stderr": ""},
    )()
    event_result = type(
        "R",
        (),
        {
            "returncode": 0,
            "stdout": "Sync\t2026-07-13T11:00:00\t2026-07-13T11:30:00\t\n",
            "stderr": "",
        },
    )()
    queried: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        script = " ".join(cmd)
        if "name of every calendar" in script:
            return list_result
        for name in ("Work", "Personal", "Exchange"):
            if f'calendar "{name}"' in script:
                queried.append(name)
        return event_result

    with patch.object(apple_calendar.subprocess, "run", side_effect=fake_run):
        events, error = get_upcoming_events(
            minutes_ahead=30,
            calendar_allowlist=["Personal"],
        )

    assert error is None
    assert queried == ["Personal"]
    assert [e["title"] for e in events] == ["Sync"]


def test_get_upcoming_events_skip_list_applies_on_allowlist() -> None:
    from escriba.calendar import apple_calendar
    from escriba.calendar.apple_calendar import (
        clear_upcoming_events_cache,
        get_upcoming_events,
    )

    clear_upcoming_events_cache()
    list_result = type(
        "R",
        (),
        {"returncode": 0, "stdout": "Work\nBirthdays\n", "stderr": ""},
    )()
    event_result = type(
        "R",
        (),
        {"returncode": 0, "stdout": "", "stderr": ""},
    )()
    queried: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        script = " ".join(cmd)
        if "name of every calendar" in script:
            return list_result
        if 'calendar "Work"' in script:
            queried.append("Work")
        if 'calendar "Birthdays"' in script:
            queried.append("Birthdays")
        return event_result

    with patch.object(apple_calendar.subprocess, "run", side_effect=fake_run):
        events, error = get_upcoming_events(
            minutes_ahead=30,
            calendar_allowlist=["Work", "Birthdays"],
        )

    assert error is None
    assert events == []
    assert queried == ["Work"]


def test_get_upcoming_events_allowlist_no_match_returns_hint_error() -> None:
    from escriba.calendar import apple_calendar
    from escriba.calendar.apple_calendar import (
        clear_upcoming_events_cache,
        get_upcoming_events,
    )

    clear_upcoming_events_cache()
    list_result = type(
        "R",
        (),
        {"returncode": 0, "stdout": "Work\nPersonal\n", "stderr": ""},
    )()

    with patch.object(apple_calendar.subprocess, "run", return_value=list_result):
        events, error = get_upcoming_events(
            minutes_ahead=30,
            calendar_allowlist=["Missing Calendar"],
        )

    assert events == []
    assert error == "no_matching_calendars"


def test_unsafe_calendar_names_are_ignored() -> None:
    from escriba.calendar.apple_calendar import (
        _filter_calendar_names,
        is_safe_calendar_name,
    )

    assert not is_safe_calendar_name('bad"name')
    assert not is_safe_calendar_name("bad\\name")
    assert not is_safe_calendar_name("bad\nname")
    assert not is_safe_calendar_name("bad\x00name")
    assert is_safe_calendar_name("Work")

    filtered = _filter_calendar_names(
        ["Work", 'bad"name'],
        calendar_allowlist=['bad"name', "Work"],
    )
    assert filtered == ["Work"]


def test_parse_calendar_names_stdout_preserves_commas() -> None:
    from escriba.calendar.apple_calendar import _parse_calendar_names_stdout

    names = _parse_calendar_names_stdout("Work\nTeam, LLC\nPersonal\n")
    assert names == ["Work", "Team, LLC", "Personal"]


def test_describe_calendars_for_settings_marks_skipped_and_selection() -> None:
    from escriba.calendar import apple_calendar
    from escriba.calendar.apple_calendar import describe_calendars_for_settings

    list_result = type(
        "R",
        (),
        {"returncode": 0, "stdout": "Work\nBirthdays\n", "stderr": ""},
    )()
    with patch.object(apple_calendar.subprocess, "run", return_value=list_result):
        entries, error = describe_calendars_for_settings(["Work"])

    assert error is None
    by_name = {entry["name"]: entry for entry in entries}
    assert by_name["Work"]["selected"] is True
    assert by_name["Work"]["skipped"] is False
    assert by_name["Birthdays"]["selected"] is False
    assert by_name["Birthdays"]["skipped"] is True


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
