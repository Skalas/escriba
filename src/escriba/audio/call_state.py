"""Pure debounce state machine for mic-activation call detection."""

from __future__ import annotations

import enum


class CallEvent(enum.Enum):
    """Edge events emitted by :class:`CallStateMachine`."""

    NONE = "none"
    CALL_STARTED = "call_started"
    CALL_ENDED = "call_ended"


class CallStateMachine:
    """
    Debounce mic-running booleans into stable call start/end events.

    The caller passes ``time.monotonic()`` as ``now`` so tests stay deterministic.
    """

    def __init__(self, start_debounce: float, stop_debounce: float) -> None:
        """
        Args:
            start_debounce: Seconds mic must stay active before CALL_STARTED.
            stop_debounce: Seconds mic must stay inactive during a call before CALL_ENDED.
        """
        self._start_debounce = start_debounce
        self._stop_debounce = stop_debounce
        self._in_call = False
        self._start_since: float | None = None
        self._stop_since: float | None = None

    @property
    def in_call(self) -> bool:
        """Whether the machine considers a call currently active."""
        return self._in_call

    def update(self, mic_running: bool, now: float) -> CallEvent:
        """
        Feed a mic-running sample and return any edge event for this tick.

        Returns:
            CALL_STARTED after mic has been continuously active for start_debounce.
            CALL_ENDED after mic has been continuously inactive for stop_debounce
            while in a call. NONE otherwise.
        """
        if not self._in_call:
            if mic_running:
                if self._start_since is None:
                    self._start_since = now
                if now - self._start_since >= self._start_debounce:
                    self._in_call = True
                    self._start_since = None
                    self._stop_since = None
                    return CallEvent.CALL_STARTED
                return CallEvent.NONE

            self._start_since = None
            return CallEvent.NONE

        if mic_running:
            self._stop_since = None
            return CallEvent.NONE

        if self._stop_since is None:
            self._stop_since = now
        if now - self._stop_since >= self._stop_debounce:
            self._in_call = False
            self._stop_since = None
            self._start_since = None
            return CallEvent.CALL_ENDED
        return CallEvent.NONE
