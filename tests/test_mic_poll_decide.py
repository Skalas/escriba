"""Unit tests for mic-activation poll decide helpers (behavior-preserving)."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from escriba.app.menubar import MicPollAction, MicPollSnapshot, TranscriberMenuBar
from escriba.audio.call_state import CallEvent


def _bare_menubar(*, start_mode: str = "prompt", cooldown_until: float = 0.0):
    """Minimal menubar stand-in for decide helpers (no rumps init)."""
    bar = object.__new__(TranscriberMenuBar)
    bar.config = SimpleNamespace(
        auto_record=SimpleNamespace(start_mode=start_mode, enabled=True)
    )
    bar._prompt_cooldown_until = cooldown_until
    bar._auto_started_session_id = "auto-1"
    bar._call_item = MagicMock()
    bar._call_item.hidden = False
    return bar


def test_decide_auto_stop_when_call_ended_and_auto_started() -> None:
    bar = _bare_menubar()
    snap = MicPollSnapshot(CallEvent.CALL_ENDED, True, True)
    assert bar._decide_mic_poll_action(snap) is MicPollAction.AUTO_STOP


def test_decide_does_not_auto_stop_hand_started() -> None:
    bar = _bare_menubar()
    snap = MicPollSnapshot(CallEvent.CALL_ENDED, True, False)
    assert bar._decide_mic_poll_action(snap) is MicPollAction.HIDE_WHILE_RECORDING


def test_decide_prompt_vs_auto_on_call_started() -> None:
    prompt_bar = _bare_menubar(start_mode="prompt")
    assert (
        prompt_bar._decide_mic_poll_action(
            MicPollSnapshot(CallEvent.CALL_STARTED, False, False)
        )
        is MicPollAction.PROMPT
    )
    auto_bar = _bare_menubar(start_mode="auto")
    assert (
        auto_bar._decide_mic_poll_action(
            MicPollSnapshot(CallEvent.CALL_STARTED, False, False)
        )
        is MicPollAction.AUTO_START
    )


def test_decide_cooldown_blocks_prompt() -> None:
    bar = _bare_menubar(cooldown_until=time.time() + 60)
    assert (
        bar._decide_mic_poll_action(
            MicPollSnapshot(CallEvent.CALL_STARTED, False, False)
        )
        is MicPollAction.COOLDOWN
    )
