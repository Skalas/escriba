"""Tests for the mic-activation call debounce state machine."""

from __future__ import annotations

from escriba.audio.call_state import CallEvent, CallStateMachine


def test_t1_sustained_on_emits_one_call_started() -> None:
    """T1: sustained mic on -> one CALL_STARTED after start debounce."""
    sm = CallStateMachine(start_debounce=3.0, stop_debounce=5.0)
    assert sm.update(False, 0.0) == CallEvent.NONE
    assert sm.update(True, 1.0) == CallEvent.NONE
    assert sm.update(True, 3.9) == CallEvent.NONE
    assert sm.update(True, 4.0) == CallEvent.CALL_STARTED
    assert sm.in_call is True
    assert sm.update(True, 4.0) == CallEvent.NONE
    assert sm.update(True, 10.0) == CallEvent.NONE


def test_t2_blip_within_start_debounce_emits_none() -> None:
    """T2: mic blip before start debounce -> NONE and timer reset."""
    sm = CallStateMachine(start_debounce=3.0, stop_debounce=5.0)
    assert sm.update(True, 0.0) == CallEvent.NONE
    assert sm.update(True, 2.0) == CallEvent.NONE
    assert sm.update(False, 2.5) == CallEvent.NONE
    assert sm.in_call is False
    assert sm.update(True, 3.0) == CallEvent.NONE
    assert sm.update(True, 5.9) == CallEvent.NONE
    assert sm.update(True, 6.0) == CallEvent.CALL_STARTED


def test_t3_sustained_off_during_call_emits_one_call_ended() -> None:
    """T3: sustained mic off during call -> one CALL_ENDED after stop debounce."""
    sm = CallStateMachine(start_debounce=1.0, stop_debounce=5.0)
    assert sm.update(True, 0.0) == CallEvent.NONE
    assert sm.update(True, 1.0) == CallEvent.CALL_STARTED
    assert sm.update(False, 2.0) == CallEvent.NONE
    assert sm.update(False, 6.9) == CallEvent.NONE
    assert sm.update(False, 7.0) == CallEvent.CALL_ENDED
    assert sm.in_call is False
    assert sm.update(False, 8.0) == CallEvent.NONE


def test_t4_off_then_on_within_stop_debounce_stays_in_call() -> None:
    """T4: brief mic drop during call -> stays in call, no CALL_ENDED."""
    sm = CallStateMachine(start_debounce=1.0, stop_debounce=5.0)
    assert sm.update(True, 0.0) == CallEvent.NONE
    assert sm.update(True, 1.0) == CallEvent.CALL_STARTED
    assert sm.update(False, 2.0) == CallEvent.NONE
    assert sm.update(False, 4.0) == CallEvent.NONE
    assert sm.update(True, 4.5) == CallEvent.NONE
    assert sm.in_call is True
    assert sm.update(False, 5.0) == CallEvent.NONE
    assert sm.update(False, 9.9) == CallEvent.NONE
    assert sm.update(False, 10.0) == CallEvent.CALL_ENDED
