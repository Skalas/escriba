"""A recording that loses one source keeps the other, and says so.

Covers the two ways a capture silently degrades on macOS: an input device
another app holds exclusively (a headset in Bluetooth call mode), and a system
tap that keeps clocking while delivering nothing but digital silence.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from escriba.app.session import (
    MIC_LOW_RMS_THRESHOLD_INT16,
    MIC_LOW_RMS_WARN_SECONDS,
    SYSTEM_SILENCE_COLD_WARN_SECONDS,
    SYSTEM_SILENCE_WARN_SECONDS,
    TranscriptionSession,
    _WARNING_PRIORITY,
    _is_built_in_mic,
)
from escriba.config import AppConfig


@pytest.fixture
def session() -> TranscriptionSession:
    base = AppConfig()
    config = dataclasses.replace(
        base, audio=dataclasses.replace(base.audio, audio_source="both")
    )
    return TranscriptionSession(config)


def _silence(session: TranscriptionSession, seconds: float) -> bytes:
    audio = session.config.audio
    return b"\x00" * int(audio.sample_rate * audio.channels * 2 * seconds)


def _tone(session: TranscriptionSession, seconds: float) -> bytes:
    audio = session.config.audio
    return b"\x01\x10" * int(audio.sample_rate * audio.channels * seconds)


def _mic_pcm(session: TranscriptionSession, seconds: float, rms_int16: float) -> bytes:
    """Build mic PCM bytes with a realistic crest factor.

    Uses Gaussian noise scaled to the desired RMS so peak/RMS ratio matches
    real mic audio (~4-9×), preventing peak-vs-RMS regressions.
    """
    audio = session.config.audio
    n_samples = int(audio.sample_rate * audio.channels * seconds)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n_samples).astype(np.float32)
    actual_rms = float(np.sqrt(np.mean(noise ** 2)))
    if actual_rms > 0:
        noise *= rms_int16 / actual_rms
    return noise.astype(np.int16).tobytes()


@pytest.fixture(autouse=True)
def _no_real_audio_hardware():
    """Keep the real machine's audio devices out of this module.

    `_start_mic_capture` calls `_default_input_shares_output_device`, which
    queries the live default input and output. On a machine where those are the
    same device — any Bluetooth headset, which is precisely the hardware this
    module is about — that shared-device path fires inside tests that never
    meant to exercise it, and the suite's result depends on what happens to be
    plugged in. It cost a real failure on 2026-09-01 with a MOMENTUM 4 connected.

    Default world: default input and output are different devices, so nothing is
    shared. Tests that need a specific device layout patch `sys.modules` inside
    their own `with` block, which takes precedence over this fixture.
    """
    neutral = _fake_sd(
        _AIRPODS_DEVICES, "Neutral Test Mic", "Neutral Test Speakers", 0
    )
    with patch.dict("sys.modules", {"sounddevice": neutral}):
        yield


# --- silent system tap (T8: warning re-arms after consume) ------------------

def test_silence_below_the_threshold_is_just_a_quiet_moment(session) -> None:
    session._track_system_silence(_silence(session, SYSTEM_SILENCE_WARN_SECONDS - 10))
    assert session.warning is None


def test_sustained_digital_silence_warns_and_rearms_after_consume(session) -> None:
    """T8: a still-true degradation keeps the banner asserted."""
    # Warm the tap first: signal, then silence, is the fault worth reporting.
    session._track_system_silence(_tone(session, 1.0))
    # First threshold crossing → warning set
    session._track_system_silence(_silence(session, SYSTEM_SILENCE_WARN_SECONDS))
    assert "No system audio" in (session.warning or "")

    # Consumer dismisses the banner
    assert session.consume_warning() is not None
    assert session.warning is None

    # Silence continues → warning re-arms on the next chunk
    session._track_system_silence(_silence(session, 1))
    assert "No system audio" in (session.warning or ""), "warning should re-arm while condition is still true"


def test_sustained_digital_silence_does_not_rearm_after_signal_recovery(session) -> None:
    """Once real signal returns the warning condition clears."""
    session._track_system_silence(_silence(session, SYSTEM_SILENCE_WARN_SECONDS))
    session.consume_warning()

    # Real signal arrives — condition resolves
    session._track_system_silence(_tone(session, 1))

    # Fresh silence starts; must not immediately warn (below threshold again)
    session._track_system_silence(_silence(session, SYSTEM_SILENCE_WARN_SECONDS - 10))
    assert session.warning is None


def test_any_real_signal_resets_the_silence_run(session) -> None:
    session._track_system_silence(_silence(session, SYSTEM_SILENCE_WARN_SECONDS - 1))
    session._track_system_silence(_tone(session, 1))
    session._track_system_silence(_silence(session, SYSTEM_SILENCE_WARN_SECONDS - 1))

    assert session.warning is None


# --- T9: healthy capture produces no warning --------------------------------

def test_healthy_system_audio_never_warns(session) -> None:
    """T9: 60s of real (non-zero) system audio must not raise any warning."""
    for _ in range(3):
        session._track_system_silence(_tone(session, 20))
    assert session.warning is None


def test_healthy_mic_rms_never_warns(session) -> None:
    """T9: 90s of mic audio above the threshold must not raise any warning."""
    high_rms = MIC_LOW_RMS_THRESHOLD_INT16 * 2
    for _ in range(3):
        session._track_mic_degradation(_mic_pcm(session, 30, high_rms))
    assert session.warning is None


# --- T7: mic RMS sub-threshold detection ------------------------------------

def test_mic_rms_below_threshold_for_short_time_no_warning(session) -> None:
    """Brief quiet mic must not trip the degradation warning."""
    low_rms = MIC_LOW_RMS_THRESHOLD_INT16 / 2
    session._track_mic_degradation(_mic_pcm(session, MIC_LOW_RMS_WARN_SECONDS - 5, low_rms))
    assert session.warning is None


def test_sustained_low_mic_rms_warns(session) -> None:
    """T7: mic at -50 dBFS for the full threshold duration raises a warning."""
    low_rms = MIC_LOW_RMS_THRESHOLD_INT16 / 2  # well below threshold (≈ -51 dBFS)
    session._track_mic_degradation(_mic_pcm(session, MIC_LOW_RMS_WARN_SECONDS, low_rms))
    assert session.warning is not None
    assert "degraded" in session.warning.lower() or "low" in session.warning.lower()


def test_mic_rms_warning_rearms_after_consume(session) -> None:
    """T8: mic degradation warning re-arms after consume if condition persists."""
    low_rms = MIC_LOW_RMS_THRESHOLD_INT16 / 2
    session._track_mic_degradation(_mic_pcm(session, MIC_LOW_RMS_WARN_SECONDS, low_rms))
    session.consume_warning()

    # More low-level mic data
    session._track_mic_degradation(_mic_pcm(session, 1, low_rms))
    assert session.warning is not None, "mic degradation warning should re-arm while condition is still true"


def test_mic_rms_warning_clears_on_recovery(session) -> None:
    """Once mic RMS returns above threshold the degraded flag is cleared."""
    low_rms = MIC_LOW_RMS_THRESHOLD_INT16 / 2
    high_rms = MIC_LOW_RMS_THRESHOLD_INT16 * 2

    session._track_mic_degradation(_mic_pcm(session, MIC_LOW_RMS_WARN_SECONDS, low_rms))
    session.consume_warning()

    # Signal recovers
    session._track_mic_degradation(_mic_pcm(session, 1, high_rms))

    # Fresh low-level data below the sustained threshold → no warning yet
    session._track_mic_degradation(_mic_pcm(session, MIC_LOW_RMS_WARN_SECONDS - 5, low_rms))
    assert session.warning is None


# --- unavailable mic --------------------------------------------------------

def test_built_in_mic_is_probed_before_continuity_devices(session) -> None:
    # W10: only built-in mics are in the fallback list; Continuity/AirPods are excluded.
    devices = [
        {"name": "Skls iPhone Microphone", "max_input_channels": 1},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        {"name": "Studio Display Speakers", "max_input_channels": 0},
        {"name": "AirPods Pro", "max_input_channels": 1},
    ]
    fake_sd = MagicMock()
    fake_sd.query_devices.side_effect = lambda index=None, **kw: (
        devices if index is None else devices[index]
    )
    fake_sd.default.device = (3, 4)  # AirPods are the default input

    with patch.dict("sys.modules", {"sounddevice": fake_sd}):
        candidates = session._candidate_mic_devices()

    # Default (None) first, then built-in Mac mic (index 1); iPhone and AirPods excluded.
    assert candidates == [None, 1]


def test_default_mic_failure_falls_back_and_names_the_substitute(session) -> None:
    opened: list[object] = []

    def flaky_open(device, callback_fn, mix_mode):
        if device is None:
            raise RuntimeError("device is held by another app")
        opened.append(device)
        session._mic_device_name = "MacBook Pro Microphone"

    with patch.object(session, "_open_mic_stream", side_effect=flaky_open), patch.object(
        session, "_candidate_mic_devices", return_value=[None, 1]
    ):
        session._start_mic_capture(mix_mode=True)

    assert opened == [1]

    fake_sd = MagicMock()
    fake_sd.query_devices.return_value = {"name": "AirPods Pro"}
    with patch.dict("sys.modules", {"sounddevice": fake_sd}):
        session._warn_if_mic_is_a_fallback()

    assert "AirPods Pro was unavailable" in (session.warning or "")
    assert "MacBook Pro Microphone" in session.warning


def test_no_usable_input_device_raises_the_original_error(session) -> None:
    with patch.object(
        session, "_open_mic_stream", side_effect=RuntimeError("first failure")
    ), patch.object(session, "_candidate_mic_devices", return_value=[None, 1, 2]):
        with pytest.raises(RuntimeError, match="first failure"):
            session._start_mic_capture(mix_mode=True)


def test_matching_mic_needs_no_warning(session) -> None:
    session._mic_device_name = "MacBook Pro Microphone"
    fake_sd = MagicMock()
    fake_sd.query_devices.return_value = {"name": "MacBook Pro Microphone"}

    with patch.dict("sys.modules", {"sounddevice": fake_sd}):
        session._warn_if_mic_is_a_fallback()

    assert session.warning is None


@pytest.mark.parametrize(
    "name, built_in",
    [
        ("MacBook Pro Microphone", True),
        ("Mac Studio Microphone", True),
        ("Skls iPhone Microphone", False),
        ("Miguel's AirPods Pro", False),
    ],
)
def test_built_in_mic_detection(name: str, built_in: bool) -> None:
    assert _is_built_in_mic(name) is built_in


# --- B1: mic degradation tracked in mic-only mode --------------------------

def test_mic_degradation_tracked_in_mic_only_mode() -> None:
    """B1: _flush_buffer must call _track_mic_degradation in mic-only mode."""
    import dataclasses
    from unittest.mock import MagicMock

    base = AppConfig()
    config = dataclasses.replace(
        base, audio=dataclasses.replace(base.audio, audio_source="mic")
    )
    mic_session = TranscriptionSession(config)
    # Simulate an active mic stream so the tracker guard passes.
    mic_session._mic_stream = MagicMock()

    low_rms = MIC_LOW_RMS_THRESHOLD_INT16 / 2
    pcm = _mic_pcm(mic_session, MIC_LOW_RMS_WARN_SECONDS, low_rms)
    mic_session._audio_buffer = bytearray(pcm)

    # _flush_buffer is the real path; the bug was mic_pcm = b"" in the non-"both" branch.
    mic_session._flush_buffer()

    assert mic_session.warning is not None, (
        "mic degradation warning must fire in mic-only mode"
    )


# --- B2: single-source sessions must not fire phantom degradation warnings ---

@pytest.fixture
def session_system() -> TranscriptionSession:
    base = AppConfig()
    config = dataclasses.replace(
        base, audio=dataclasses.replace(base.audio, audio_source="system")
    )
    return TranscriptionSession(config)


@pytest.fixture
def session_mic_only() -> TranscriptionSession:
    base = AppConfig()
    config = dataclasses.replace(
        base, audio=dataclasses.replace(base.audio, audio_source="mic")
    )
    return TranscriptionSession(config)


def test_system_only_no_phantom_mic_warning(session_system) -> None:
    """_flush_buffer guard: _track_mic_degradation not called in system-only mode."""
    session_system._audio_buffer = bytearray(_silence(session_system, 1.0))
    with patch.object(session_system, "_track_mic_degradation") as mock_mic, \
         patch.object(session_system, "_track_system_silence"):
        session_system._flush_buffer()
    mock_mic.assert_not_called()


def test_mic_only_no_phantom_system_warning(session_mic_only) -> None:
    """_flush_buffer guard: _track_system_silence not called in mic-only mode."""
    session_mic_only._audio_buffer = bytearray(_silence(session_mic_only, 1.0))
    with patch.object(session_mic_only, "_track_system_silence") as mock_sys, \
         patch.object(session_mic_only, "_track_mic_degradation"):
        session_mic_only._flush_buffer()
    mock_sys.assert_not_called()


def test_both_mode_mic_failed_no_stacked_mic_warning(session) -> None:
    """In 'both' mode, if _mic_stream is None (mic never started), mic tracker is skipped."""
    session._system_buffer = bytearray(_silence(session, 1.0))
    session._mic_buffer = bytearray()
    session._mic_stream = None
    with patch.object(session, "_track_mic_degradation") as mock_mic, \
         patch.object(session, "_track_system_silence"):
        session._flush_buffer()
    mock_mic.assert_not_called()


# --- B3: [tap] dead token parsed in _drain_stderr ---------------------------

def test_drain_stderr_calls_on_tap_dead_callback() -> None:
    """_drain_stderr calls on_tap_dead exactly once when it sees '[tap] dead'."""
    from unittest.mock import MagicMock
    from escriba.audio.screen_capture import ScreenCaptureAudioCapture

    dead_calls: list[int] = []
    cap = MagicMock()
    cap.on_tap_dead = lambda: dead_calls.append(1)
    cap._callbacks_disabled = False  # M6: must be False for callbacks to fire
    cap.process.stderr = iter([
        b"[tap] rebuilding after sustained silence\n",
        b"[tap] dead: 5 consecutive rebuild failures\r\n",
        b"[tap] retrying at 60s cadence\n",
    ])

    ScreenCaptureAudioCapture._drain_stderr(cap)

    assert dead_calls == [1], "on_tap_dead must fire exactly once for '[tap] dead' line"


def test_drain_stderr_no_false_positive() -> None:
    """Lines that merely mention 'dead' but lack the exact prefix do not fire callback."""
    from unittest.mock import MagicMock
    from escriba.audio.screen_capture import ScreenCaptureAudioCapture

    dead_calls: list[int] = []
    cap = MagicMock()
    cap.on_tap_dead = lambda: dead_calls.append(1)
    cap._callbacks_disabled = False
    cap.process.stderr = iter([
        b"[tap] tapDead flag cleared\n",
        b"[tap] rebuild after dead\n",  # "dead" not at prefix
    ])

    ScreenCaptureAudioCapture._drain_stderr(cap)

    assert dead_calls == [], "no on_tap_dead call for lines that don't start with '[tap] dead'"


# --- B2 guard: mic-only + _mic_stream is None --------------------------------

def test_mic_only_mic_stream_none_no_mic_tracker(session_mic_only) -> None:
    """_flush_buffer guard: mic tracker skipped in mic-only mode if _mic_stream is None."""
    session_mic_only._audio_buffer = bytearray(_silence(session_mic_only, 1.0))
    session_mic_only._mic_stream = None
    with patch.object(session_mic_only, "_track_mic_degradation") as mock_mic, \
         patch.object(session_mic_only, "_track_system_silence"):
        session_mic_only._flush_buffer()
    mock_mic.assert_not_called()


def test_quiet_start_does_not_warn_before_the_cold_threshold(session) -> None:
    """A tap that has never produced signal is probably an idle output device.

    Observed 2026-09-01: a real session warned "nothing is being recorded" 123s
    in, because nobody had played anything yet. Two quiet minutes at the start
    of a recording is the normal case and must not raise an alarm.
    """
    session._track_system_silence(_silence(session, SYSTEM_SILENCE_WARN_SECONDS + 5))
    assert session.warning is None, (
        "warned about a tap that was never expected to have audio yet"
    )


def test_quiet_start_does_warn_once_the_cold_threshold_passes(session) -> None:
    """Five minutes into a call with nothing on the system tap is worth saying."""
    session._track_system_silence(
        _silence(session, SYSTEM_SILENCE_COLD_WARN_SECONDS + 1)
    )
    assert "No system audio" in (session.warning or "")
    assert "5 min" in session.warning, session.warning


def test_a_warmed_tap_going_silent_warns_at_the_warm_threshold(session) -> None:
    """The real fault: the tap produced audio, then stopped."""
    session._track_system_silence(_tone(session, 1.0))
    session._track_system_silence(_silence(session, SYSTEM_SILENCE_WARN_SECONDS))
    assert "No system audio" in (session.warning or "")
    assert "2 min" in session.warning, session.warning


# --- W3: system silence warning text branches on mic availability ------------

def test_system_silence_warning_with_mic_says_only_mic(session) -> None:
    """W3: when mic is capturing, warning says 'only your microphone is being recorded'."""
    from unittest.mock import MagicMock
    session._mic_stream = MagicMock()
    session._track_system_silence(_tone(session, 1.0))  # warm: the tap worked once
    session._track_system_silence(_silence(session, SYSTEM_SILENCE_WARN_SECONDS))
    assert session.warning is not None
    assert "only your microphone" in session.warning


def test_system_silence_warning_without_mic_says_nothing_recorded(session_system) -> None:
    """W3: in system-only mode, warning says 'nothing is being recorded'."""
    session_system._track_system_silence(_tone(session_system, 1.0))
    session_system._track_system_silence(_silence(session_system, SYSTEM_SILENCE_WARN_SECONDS))
    assert session_system.warning is not None
    assert "nothing is being recorded" in session_system.warning


# --- W4: tap_dead flag re-arms and clears ------------------------------------

def test_tap_dead_rearms_on_each_flush(session) -> None:
    """W4: while _tap_dead is set, each _flush_buffer re-asserts the warning."""
    session.is_active = True  # re-arm guard requires an active session
    session._tap_dead = True
    session._flush_buffer()
    assert session.warning is not None
    assert "audio chain failed" in session.warning

    # Consumer dismisses; next flush re-arms.
    session.consume_warning()
    assert session.warning is None
    session._flush_buffer()
    assert session.warning is not None, "tap_dead warning must re-arm after dismiss"


def test_tap_dead_clears_on_non_silent_system_chunk(session) -> None:
    """W4: real system audio clears the _tap_dead flag and removes the warning."""
    session._tap_dead = True
    session._set_warning(
        "tap_dead",
        "System audio capture has stopped — the audio chain failed repeatedly. "
        "Try stopping and restarting the recording.",
    )
    session._track_system_silence(_tone(session, 1))
    assert not session._tap_dead
    assert session.consume_warning() is None, "tap_dead warning must be cleared on signal recovery"


# --- W5: _on_tap_dead is a no-op after session stopped ----------------------

def test_on_tap_dead_ignored_after_stop(session) -> None:
    """W5: _on_tap_dead must not set a warning when is_active is False."""
    session.is_active = False
    session._on_tap_dead()
    assert session.warning is None
    assert not session._tap_dead


def test_tap_dead_warning_cleared_by_stop(session) -> None:
    """M7: stop() clears any tap_dead (or other degradation) warning left by final flush.

    Without the M7 fix, _tap_dead=True → final flush re-arms warning → the next
    status poll after stop shows 'audio chain failed' for a recording that already ended.
    """
    # Latch the tap_dead flag and set a warning (as if the tap died during recording).
    session.is_active = True
    session._on_tap_dead()
    assert session.warning is not None, "precondition: warning is set"

    # stop() sets is_active=False first, so the re-arm guard in _flush_buffer fires.
    # The post-flush warning clear in stop() then removes any residual warnings.
    session.stop()

    assert session.warning is None, "stop() must clear tap_dead warning (M7)"


# --- M4: on_tap_dead wiring tested at the real construction site ------------

def test_on_tap_dead_wired_to_session() -> None:
    """M4: session.start() passes self._on_tap_dead to ScreenCaptureAudioCapture.

    The regression this catches: removing or changing the on_tap_dead= kwarg at
    session.py line ~242. The previous test was tautological — it supplied the
    kwarg itself and had a truthiness guard that let it pass even if no constructor
    was ever called. This version patches the class at its source module (so the
    function-scope 'from ... import' inside start() picks up the mock) and calls
    the real start() path, asserting unconditionally.
    """
    import dataclasses
    from unittest.mock import MagicMock, patch

    base = AppConfig()
    config = dataclasses.replace(
        base, audio=dataclasses.replace(base.audio, audio_source="system")
    )
    sess = TranscriptionSession(config)
    captured_kwargs: list[dict] = []

    mock_instance = MagicMock()
    mock_instance.start.return_value = True

    def cap_constructor(**kwargs):
        captured_kwargs.append(kwargs)
        return mock_instance

    # Patch at the source module so the function-scope import in start() gets the mock.
    with patch("escriba.audio.screen_capture.ScreenCaptureAudioCapture",
               side_effect=cap_constructor), \
         patch("escriba.app.session._build_transcriber", return_value=MagicMock()), \
         patch.object(sess, "_open_audio_file"), \
         patch.object(sess, "_process_loop"):
        sess.start()

    assert len(captured_kwargs) == 1, "ScreenCaptureAudioCapture must be constructed exactly once"
    # Unconditional — dropping the kwarg makes this fail immediately.
    # Use == not is: bound methods create a new object on each attribute access,
    # so `==` (which compares __self__ and __func__) is the correct identity check.
    assert captured_kwargs[0]["on_tap_dead"] == sess._on_tap_dead, (
        "on_tap_dead kwarg must be session._on_tap_dead"
    )


# --- B1: peek_warning_item returns highest-priority warning, not oldest -------

def test_peek_warning_item_severity_order(session) -> None:
    """B1: peek_warning_item must return tap_dead over direct, regardless of insertion order.

    Without _WARNING_PRIORITY, next(iter(_warnings)) returns the first-inserted key.
    With direct inserted before tap_dead (as start() does), insertion-order would
    always return direct — starving tap_dead for the entire recording.

    This test FAILS against the old insertion-order implementation:
        peek_warning_item() → ("direct", ...) but assert expects ("tap_dead", ...)
    """
    # direct inserted first (mirrors start() setting it before tap dies)
    session._set_warning("direct", "Microphone unavailable — recording system audio only")
    session._set_warning("tap_dead", "System audio capture has stopped")

    # peek must surface tap_dead despite direct being first-inserted
    first = session.peek_warning_item()
    assert first is not None
    assert first[0] == "tap_dead", (
        f"B1: expected tap_dead but got {first[0]!r} — "
        "insertion-order peek permanently starves the most urgent warning"
    )

    # second call must return the same result (peek does not consume)
    second = session.peek_warning_item()
    assert second == first, "peek_warning_item must not consume the warning"

    # direct is still reachable after tap_dead is consumed
    session._clear_warning("tap_dead")
    remaining = session.peek_warning_item()
    assert remaining is not None and remaining[0] == "direct"

    # _WARNING_PRIORITY itself: tap_dead must outrank every other key
    assert _WARNING_PRIORITY[0] == "tap_dead"
    assert _WARNING_PRIORITY.index("tap_dead") < _WARNING_PRIORITY.index("direct")


# --- self-inflicted Bluetooth profile switch --------------------------------
#
# Escriba caused the failure this sprint recovers from: pressing record opened
# the AirPods microphone, macOS dropped the shared link to HFP, and the system
# tap built 103ms earlier against the A2DP format delivered zeros for the rest
# of the session. Prevention beats recovery — don't open a Bluetooth mic that
# is also the device we are playing through.

_AIRPODS_DEVICES = [
    {"name": "Skls iPhone Microphone", "max_input_channels": 1},
    {"name": "MacBook Pro Microphone", "max_input_channels": 1},
    {"name": "Miguel's AirPods Pro", "max_input_channels": 1},
]


def _fake_sd(devices, default_input, default_output, default_index):
    """A sounddevice stand-in that answers both index and kind lookups."""
    fake = MagicMock()

    def query(index=None, kind=None, **kw):
        if kind == "input":
            return {"name": default_input}
        if kind == "output":
            return {"name": default_output}
        return devices if index is None else devices[index]

    fake.query_devices.side_effect = query
    fake.default.device = (default_index, 9)
    return fake


def test_shared_bluetooth_mic_is_demoted_below_the_built_in(session) -> None:
    """Ordering only. Whether to demote is the caller's decision (T11)."""
    fake = _fake_sd(
        _AIRPODS_DEVICES, "Miguel's AirPods Pro", "Miguel's AirPods Pro", 2
    )
    with patch.dict("sys.modules", {"sounddevice": fake}):
        candidates = session._candidate_mic_devices(demote_default=True)

    # Built-in mic first; the default stays last so a Mac without one still records.
    assert candidates == [1, None]
    assert session.warning is None, (
        "_candidate_mic_devices must not touch session state — the banner is "
        "only earned once a device is actually open"
    )


def test_banner_names_the_device_that_actually_opened(session) -> None:
    """T11: the regression that motivated moving the announcement.

    The built-in mic is preferred but fails to open, so capture falls through
    to the AirPods. A banner claiming the built-in mic is recording would be
    plainly false, and it used to suppress the correcting message too.
    """
    fake = _fake_sd(
        _AIRPODS_DEVICES, "Miguel's AirPods Pro", "Miguel's AirPods Pro", 2
    )
    opened: list[object] = []

    def open_stream(device, cb, mix):
        if device == 1:
            raise OSError("built-in mic busy")
        opened.append(device)
        session._mic_device_name = "Miguel's AirPods Pro"

    with patch.dict("sys.modules", {"sounddevice": fake}), patch.object(
        session, "_open_mic_stream", side_effect=open_stream
    ):
        session._start_mic_capture(mix_mode=True)

    assert opened == [None], "must still record rather than give up"
    with session._warnings_lock:
        assert "mic_shared_output" not in session._warnings, (
            "claimed the built-in mic while recording the AirPods"
        )


def test_banner_names_the_built_in_mic_when_it_does_open(session) -> None:
    fake = _fake_sd(
        _AIRPODS_DEVICES, "Miguel's AirPods Pro", "Miguel's AirPods Pro", 2
    )

    def open_stream(device, cb, mix):
        session._mic_device_name = _AIRPODS_DEVICES[device]["name"]

    with patch.dict("sys.modules", {"sounddevice": fake}), patch.object(
        session, "_open_mic_stream", side_effect=open_stream
    ):
        session._start_mic_capture(mix_mode=True)

    warning = session.warning or ""
    assert "AirPods" in warning and "MacBook Pro Microphone" in warning, warning


def test_record_opens_the_built_in_mic_when_output_is_bluetooth(session) -> None:
    """The real path: _start_mic_capture must never reach the AirPods first."""
    opened: list[object] = []
    fake = _fake_sd(
        _AIRPODS_DEVICES, "Miguel's AirPods Pro", "Miguel's AirPods Pro", 2
    )

    with patch.dict("sys.modules", {"sounddevice": fake}), patch.object(
        session, "_open_mic_stream", side_effect=lambda d, cb, m: opened.append(d)
    ):
        session._start_mic_capture(mix_mode=True)

    assert opened == [1], (
        "mix_mode capture opened the shared Bluetooth mic — this is the "
        "profile switch that silences the tap"
    )


def test_mic_only_session_keeps_the_bluetooth_mic(session) -> None:
    """No tap to protect, so proximity wins: the headset mic is the better take."""
    opened: list[object] = []
    fake = _fake_sd(
        _AIRPODS_DEVICES, "Miguel's AirPods Pro", "Miguel's AirPods Pro", 2
    )

    with patch.dict("sys.modules", {"sounddevice": fake}), patch.object(
        session, "_open_mic_stream", side_effect=lambda d, cb, m: opened.append(d)
    ):
        session._start_mic_capture(mix_mode=False)

    assert opened == [None]
    assert session.warning is None


def test_shared_output_without_a_built_in_mic_still_records(session) -> None:
    devices = [{"name": "Miguel's AirPods Pro", "max_input_channels": 1}]
    fake = _fake_sd(devices, "Miguel's AirPods Pro", "Miguel's AirPods Pro", 0)

    def open_stream(device, cb, mix):
        session._mic_device_name = "Miguel's AirPods Pro"

    with patch.dict("sys.modules", {"sounddevice": fake}), patch.object(
        session, "_open_mic_stream", side_effect=open_stream
    ):
        session._start_mic_capture(mix_mode=True)

    with session._warnings_lock:
        assert "mic_shared_output" not in session._warnings, (
            "no alternative exists, so there is no demotion to announce"
        )


def test_separate_input_and_output_devices_are_left_alone(session) -> None:
    """Distinct input and output devices: no demotion, no banner."""
    fake = _fake_sd(
        _AIRPODS_DEVICES, "Miguel's AirPods Pro", "MacBook Pro Speakers", 2
    )
    opened: list[object] = []

    with patch.dict("sys.modules", {"sounddevice": fake}), patch.object(
        session, "_open_mic_stream", side_effect=lambda d, cb, m: opened.append(d)
    ):
        session._start_mic_capture(mix_mode=True)

    assert opened == [None], "the user's chosen input must be tried first"
    with session._warnings_lock:
        assert "mic_shared_output" not in session._warnings


def test_keep_bluetooth_playback_off_uses_the_headset_mic(session) -> None:
    """The setting is the opt-out: use the AirPods mic and accept HFP."""
    session.config = dataclasses.replace(
        session.config,
        audio=dataclasses.replace(
            session.config.audio, keep_bluetooth_playback=False
        ),
    )
    opened: list[object] = []
    fake = _fake_sd(
        _AIRPODS_DEVICES, "Miguel's AirPods Pro", "Miguel's AirPods Pro", 2
    )

    with patch.dict("sys.modules", {"sounddevice": fake}), patch.object(
        session, "_open_mic_stream", side_effect=lambda d, cb, m: opened.append(d)
    ):
        session._start_mic_capture(mix_mode=True)

    assert opened == [None]
    assert session.warning is None


def test_deliberate_demotion_is_not_reported_as_unavailable(session) -> None:
    """The two warnings contradict each other; the deliberate one wins."""
    session._set_warning("mic_shared_output", "recording with the built-in mic")
    session._mic_device_name = "MacBook Pro Microphone"

    fake = MagicMock()
    fake.query_devices.return_value = {"name": "Miguel's AirPods Pro"}
    with patch.dict("sys.modules", {"sounddevice": fake}):
        session._warn_if_mic_is_a_fallback()

    with session._warnings_lock:
        assert "mic_fallback" not in session._warnings
