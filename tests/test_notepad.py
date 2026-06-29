"""Tests T53.x — Live notepad / user_notes feature."""
from __future__ import annotations

from pathlib import Path

import pytest

from escriba.app.database import Database
from escriba.app.session import _build_custom_prompt
from escriba.config import DEFAULT_SYSTEM_PROMPT


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


# T53.1 — Round-trip persistence
def test_user_notes_round_trip(db: Database) -> None:
    """save_user_notes → get_session returns user_notes."""
    session_id = db.create_session(name="Test Session")
    db.save_user_notes(session_id, "My context notes")
    session = db.get_session(session_id)
    assert session is not None
    assert session["user_notes"] == "My context notes"


# T53.2 — Notes injection into DEFAULT_SYSTEM_PROMPT
def test_user_notes_injected_into_default_prompt() -> None:
    """{user_notes} in DEFAULT_SYSTEM_PROMPT causes notes to appear in output."""
    result = _build_custom_prompt(
        transcript="Hello world",
        prompt="Summarize",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_notes="my notes",
    )
    assert "my notes" in result


# T53.3 — Custom prompt without {user_notes} still includes notes as preamble
def test_user_notes_prepended_when_placeholder_missing() -> None:
    """When the template has no {user_notes}, notes are prepended as XML block."""
    custom_template = "<transcript>\n{transcript}\n</transcript>\n<task>\n{prompt}\n</task>"
    result = _build_custom_prompt(
        transcript="Hello",
        prompt="Do something",
        system_prompt=custom_template,
        user_notes="important context",
    )
    assert "important context" in result
    assert result.startswith("<user_notes>")


# T53.4 — Custom prompt without {user_notes} — no KeyError raised
def test_custom_prompt_without_placeholder_no_error() -> None:
    """Template with no {user_notes} and non-empty user_notes must not raise."""
    custom_template = "<transcript>\n{transcript}\n</transcript>\n<task>\n{prompt}\n</task>"
    try:
        _build_custom_prompt(
            transcript="Hello",
            prompt="Do something",
            system_prompt=custom_template,
            user_notes="context",
        )
    except KeyError:
        pytest.fail("_build_custom_prompt raised KeyError with custom template")


# T53.6 — No custom prompt still routes user_notes into the LLM call
def test_generate_notes_no_prompt_honors_user_notes(tmp_path: Path) -> None:
    """generate_notes(prompt=None) with persisted user_notes calls _generate_custom_notes.

    Pre-fix: the else-branch called generate_summary() directly, so user_notes never
    reached the LLM. Post-fix: if user_notes exist, routing goes through
    _generate_custom_notes which embeds them via _build_custom_prompt.
    This assertion fails against the pre-fix code because _generate_custom_notes is
    never called there and `captured` stays empty.
    """
    from unittest.mock import MagicMock, patch

    import escriba.app.session as session_module
    from escriba.app.database import Database
    from escriba.config import AppConfig

    db = Database(tmp_path / "test.db")
    session_id = db.create_session(name="Test")
    db.save_user_notes(session_id, "critical context")

    cfg = AppConfig()
    sess = session_module.TranscriptionSession(cfg, database=db)
    sess.db_session_id = session_id
    sess.transcriber = MagicMock()
    sess.transcriber.get_full_transcript.return_value = "Hello world"

    captured: dict = {}

    def fake_generate_custom_notes(
        transcript, prompt, model, system_prompt=None, user_notes=""
    ):
        captured["user_notes"] = user_notes
        return "notes"

    with patch.object(session_module, "_generate_custom_notes", side_effect=fake_generate_custom_notes):
        result = sess.generate_notes(prompt=None)

    db.close()
    assert result == "notes", "generate_notes should have returned the mocked notes"
    assert "critical context" in captured.get("user_notes", ""), (
        "_generate_custom_notes was not called with user_notes — "
        "pre-fix code routes to generate_summary() instead"
    )


# T53.5 — DOM test: live-notepad textarea and session-user-notes-card are in index.html
def test_dom_elements_present_in_index_html() -> None:
    """index.html contains the live-notepad textarea and the unified notes card elements."""
    index_path = (
        Path(__file__).parent.parent
        / "src"
        / "escriba"
        / "app"
        / "static"
        / "index.html"
    )
    content = index_path.read_text(encoding="utf-8")
    assert 'id="live-notepad"' in content, "live-notepad textarea missing from index.html"
    assert 'id="notes-rendered"' in content, "notes-rendered div missing from index.html"
    assert 'id="session-notes-input"' in content, "session-notes-input missing from index.html"
