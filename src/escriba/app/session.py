"""Transcription session management for the menu bar app."""

from __future__ import annotations

import logging
import math
import os
import struct
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cap live PCM buffer at this multiple of one transcription chunk.
AUDIO_BUFFER_CAP_FACTOR = 2
AUDIO_BUFFER_OVERFLOW_LOG_INTERVAL_S = 5.0
# How long an unbroken run of effectively-silent audio from the system tap counts
# as a broken capture. "Effectively silent" means peak amplitude below
# SYSTEM_SIGNAL_PEAK_THRESHOLD_INT16, low enough to ignore thermal noise but high
# enough to catch a dead tap.
SYSTEM_SILENCE_WARN_SECONDS = 120.0
# Cold variant: the tap has never produced signal in this session. Two quiet
# minutes at the start of a recording is the normal case, not a fault — nobody
# has played anything yet. Observed 2026-09-01: a session warned "nothing is
# being recorded" 123s in, purely because playback had not started. The old
# Swift design carried this warm/cold distinction as `sawSignal` and the
# 2026-08-22 rewrite dropped it along with silence-triggered rebuild; the
# distinction was the good half.
SYSTEM_SILENCE_COLD_WARN_SECONDS = 300.0
# Any per-chunk peak below this on the system tap is treated as silence for
# degradation tracking. -60 dBFS ≈ 32 on int16 scale.
SYSTEM_SIGNAL_PEAK_THRESHOLD_INT16: float = 32.0
# A mic whose per-chunk RMS amplitude stays below this for an extended time is
# almost certainly degraded. -45 dBFS ≈ 184 on int16 scale. RMS (not peak) is
# used so a brief burst of loud noise inside a long quiet window does not falsely
# clear the degraded flag — only sustained speech clears it.
MIC_LOW_RMS_THRESHOLD_INT16: float = 184.0
_MIC_THRESHOLD_DBFS: int = round(20 * math.log10(MIC_LOW_RMS_THRESHOLD_INT16 / 32767))
MIC_LOW_RMS_WARN_SECONDS = 60.0

# How long stop() waits for the audio-processing thread. Its remaining work is
# bounded: at most the buffer cap plus a flush already in flight, so roughly
# three chunks of audio to transcribe. A loaded machine transcribes at about
# half of realtime, hence twice that in wall-clock. A single deadline (rather
# than polling for progress) is the only honest measure here — segments land
# only when a whole chunk finishes decoding, so an in-flight chunk is
# indistinguishable from a deadlock while it runs. If the thread is still alive
# when the wait ends, the main thread must NOT touch the transcriber or WAV
# writer concurrently (see stop()).
PROCESS_JOIN_CHUNK_FACTOR = 6
PROCESS_JOIN_MIN_TIMEOUT_S = 60.0
# How long stop() waits for an in-flight preliminary title generation before
# refining. Two concurrent mlx-lm generations crash Metal, so if the thread is
# still alive after this, the refine pass is skipped.
TITLE_JOIN_TIMEOUT_S = 30.0


# macOS names the built-in input after the machine model ("MacBook Pro
# Microphone", "Mac Studio Microphone"), which is the only handle Core Audio
# gives us to tell it apart from Continuity and USB inputs.
_BUILT_IN_MIC_MARKERS = ("macbook", "imac", "mac mini", "mac studio", "mac pro", "built-in")


def _is_built_in_mic(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _BUILT_IN_MIC_MARKERS)


class _SessionStartAborted(Exception):
    """Raised internally to funnel every start() failure through cleanup.

    The human-facing reason is set on ``session.error`` before raising; this
    only steers control flow to ``_abort_start()``.
    """


# B1: severity order for peek_warning_item. tap_dead is most urgent (audio stopped
# entirely); direct/mic_fallback are lowest (set-once informational at start()).
# Unknown keys fall back to insertion order via the loop-exit path in peek_warning_item.
_WARNING_PRIORITY = (
    "tap_dead", "system", "mic", "mic_shared_output", "mic_fallback", "direct",
)


class TranscriptionSession:
    """Manages a single transcription session: capture + transcribe + notes."""

    def __init__(self, config, database=None):
        from escriba.config import AppConfig

        self.config: AppConfig = config
        self.db = database
        self.transcriber = None
        self.screen_capture = None
        self._mic_stream = None
        self._mic_device_name: str | None = None
        self._audio_buffer = bytearray()
        self._system_buffer = bytearray()
        self._mic_buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._process_thread = None
        self.is_active = False
        self.start_time: datetime | None = None
        self.session_id: str | None = None
        self.db_session_id: str | None = None
        self._last_segment_count: int = 0
        self.output_dir = Path("transcripts")
        self.error: str | None = None
        self._warnings: dict[str, str] = {}
        self._warnings_lock = threading.Lock()  # W1: guards _warnings across threads
        self._silent_system_seconds: float = 0.0
        self._system_degraded: bool = False
        self._last_system_signal_ts: float = 0.0   # B4: wall-clock of last non-silent system pcm
        self._system_tracking_started: float = 0.0  # B4: when _track_system_silence first called
        self._mic_rms_low_seconds: float = 0.0
        self._mic_degraded: bool = False
        self._last_mic_signal_ts: float = 0.0   # B4: wall-clock of last healthy mic rms
        self._mic_tracking_started: float = 0.0  # B4: when _track_mic_degradation first called
        self._tap_dead: bool = False  # W4: latched when Swift tap emits '[tap] dead'
        self._audio_file: Path | None = None
        self._audio_writer: wave.Wave_write | None = None
        self.detected_app: str | None = None
        self.initial_name: str | None = None
        self._title_generated: bool = False
        self._title_refined: bool = False
        self._title_thread: threading.Thread | None = None
        self._last_buffer_overflow_log: float = 0.0

    @property
    def warning(self) -> str | None:
        """First pending warning across all sources, or None."""
        with self._warnings_lock:
            return next(iter(self._warnings.values()), None)

    @warning.setter
    def warning(self, value: str | None) -> None:
        with self._warnings_lock:
            if value is None:
                self._warnings.pop("direct", None)
            else:
                self._warnings["direct"] = value

    def _set_warning(self, key: str, msg: str) -> None:
        with self._warnings_lock:
            self._warnings[key] = msg

    def _clear_warning(self, key: str) -> None:
        with self._warnings_lock:
            self._warnings.pop(key, None)

    def _open_audio_file(self):
        """Open a WAV file to record the session audio."""
        audio_dir = Path.home() / "Library" / "Application Support" / "Escriba" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.db_session_id or self.session_id}.wav"
        self._audio_file = audio_dir / filename
        try:
            self._audio_writer = wave.open(str(self._audio_file), "wb")
            self._audio_writer.setnchannels(self.config.audio.channels)
            self._audio_writer.setsampwidth(2)  # 16-bit = 2 bytes
            self._audio_writer.setframerate(self.config.audio.sample_rate)
            logger.info("Audio recording to: %s", self._audio_file)
        except Exception as e:
            logger.error("Failed to open audio file: %s", e)
            self._audio_writer = None

    def _close_audio_file(self):
        """Close the WAV file and store the path in DB."""
        if self._audio_writer:
            try:
                self._audio_writer.close()
            except Exception as e:
                logger.error("Failed to close audio file: %s", e)
            self._audio_writer = None
        if self._audio_file and self._audio_file.exists() and self.db and self.db_session_id:
            self.db.update_audio_path(self.db_session_id, str(self._audio_file))

    def start(self):
        if self.is_active:
            return

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self._stop_event.clear()
        self._audio_buffer = bytearray()
        self._system_buffer = bytearray()
        self._mic_buffer = bytearray()
        self._last_segment_count = 0
        self.error = None
        with self._warnings_lock:
            self._warnings.clear()
        self._silent_system_seconds = 0.0
        self._system_degraded = False
        self._last_system_signal_ts = 0.0
        self._system_tracking_started = 0.0
        self._mic_rms_low_seconds = 0.0
        self._mic_degraded = False
        self._last_mic_signal_ts = 0.0
        self._mic_tracking_started = 0.0
        self._last_buffer_overflow_log = 0.0

        # Create DB session
        if self.db:
            backend = self.config.streaming.backend
            model_size = self.config.streaming.model_size
            language = self.config.streaming.language
            name = (
                (self.initial_name or "").strip()
                or f"Session {self.start_time.strftime('%Y-%m-%d %H:%M')}"
            )
            self.db_session_id = self.db.create_session(
                name=name, model=model_size, language=language, backend=backend,
            )

        # Open WAV file for audio recording
        self._open_audio_file()

        # Everything below opens resources (transcriber, subprocess, mic stream)
        # that must be released if any later step fails. Funnel every failure
        # through _abort_start() so we never leak a Swift child / open WAV handle
        # or leave the DB row 'active' (the caller discards a session whose
        # is_active is False without calling stop()).
        try:
            backend = self.config.streaming.backend
            model_size = self.config.streaming.model_size
            language = self.config.streaming.language
            logger.info(
                "Session config: backend=%s, model=%s, language=%s",
                backend, model_size, language,
            )

            try:
                self.transcriber = _build_transcriber(self.config, realtime_output=True)
            except Exception as e:
                self.error = "Failed to load transcription model"
                logger.error("Failed to load model: %s", e, exc_info=True)
                raise _SessionStartAborted from e

            audio_source = self.config.audio.audio_source
            logger.info("Audio source: %s", audio_source)

            # Start system audio (for "system" and "both" modes)
            if audio_source in ("system", "both"):
                try:
                    from escriba.audio.screen_capture import (
                        ScreenCaptureAudioCapture,
                    )

                    self.screen_capture = ScreenCaptureAudioCapture(
                        sample_rate=self.config.audio.sample_rate,
                        channels=self.config.audio.channels,
                        audio_callback=self._on_system_audio if audio_source == "both" else self._on_audio_data,
                        on_tap_dead=self._on_tap_dead,
                    )
                except ImportError as e:
                    self.error = (
                        "Swift audio-capture CLI not available. "
                        "Build with: cd swift-audio-capture && swift build -c release"
                    )
                    logger.error(self.error)
                    raise _SessionStartAborted from e
                if not self.screen_capture.start():
                    self.error = "Failed to start audio capture. Check permissions."
                    raise _SessionStartAborted

            # Start mic capture (for "mic" and "both" modes)
            if audio_source in ("mic", "both"):
                try:
                    self._start_mic_capture(mix_mode=audio_source == "both")
                except Exception as e:
                    logger.error("Failed to start microphone capture: %s", e, exc_info=True)
                    # In "both" mode the system tap is already running, so a mic
                    # failure costs one of two sources — not the recording. A
                    # headset in Bluetooth call mode routinely holds the mic
                    # exclusively; dropping the whole session over it loses the
                    # very call the user meant to capture.
                    if audio_source == "mic":
                        self.error = "Failed to start microphone capture. Check permissions."
                        raise _SessionStartAborted from e
                    self._set_warning(
                        "direct",
                        "Microphone unavailable — recording system audio only. "
                        "Another app may be holding it.",
                    )
                else:
                    self._warn_if_mic_is_a_fallback()

            # Start processing thread
            self._process_thread = threading.Thread(
                target=self._process_loop, daemon=True
            )
            self._process_thread.start()
            self.is_active = True
            logger.info("Session started: %s", self.session_id)

        except _SessionStartAborted:
            self._abort_start()
        except Exception as e:
            self.error = self.error or "Failed to start recording"
            logger.error("Unexpected session start failure: %s", e, exc_info=True)
            self._abort_start()

    def _abort_start(self):
        """Release everything start() opened and mark the DB row errored.

        Safe to call after a partial start: each step is guarded so it runs
        whether or not the corresponding resource was actually acquired.
        """
        if self._mic_stream is not None:
            try:
                self._close_mic_stream_safely()
            except Exception:
                logger.warning("Failed to close mic stream during abort", exc_info=True)
        if self.screen_capture:
            try:
                self.screen_capture.stop()
            except Exception:
                logger.warning("Failed to stop screen capture during abort", exc_info=True)
            self.screen_capture = None
        self._close_audio_file()
        if self.db and self.db_session_id:
            try:
                self.db.stop_session(self.db_session_id, status="error")
            except Exception:
                logger.warning("Failed to mark session errored during abort", exc_info=True)

    @staticmethod
    def _run_cleanup_step(name: str, step) -> bool:
        """Run one finalization step, isolating its failure from the others.

        stop() must complete every cleanup step (buffer flush, WAV close,
        export, DB finalization) even if an earlier one raises, so a capture
        teardown exception can never leave a session half-written. Returns True
        on success, False if the step raised.
        """
        try:
            step()
            return True
        except Exception:
            logger.exception("Session stop step failed: %s", name)
            return False

    def stop(self):
        if not self.is_active:
            return

        self.is_active = False
        self._stop_event.set()

        # Tear down capture sources first. Each runs independently so one
        # failure cannot skip the rest of finalization.
        if self._mic_stream is not None:
            self._run_cleanup_step("mic stream close", self._close_mic_stream_safely)
        if self.screen_capture:
            self._run_cleanup_step("screen capture stop", self.screen_capture.stop)

        # Join the audio-processing thread. If it does NOT finish in time it may
        # still be reading the buffers / driving the transcriber / writing the
        # WAV, so the main thread must not flush the buffer or close the WAV
        # writer concurrently — that races on shared state and can corrupt the
        # recording. We still finalize the DB row so the session isn't left
        # ACTIVE forever.
        process_thread_finished = self._join_process_thread()
        if not process_thread_finished:
            self.error = self.error or "Recording did not stop cleanly"

        if process_thread_finished:
            self._run_cleanup_step("final buffer flush", self._flush_buffer)
            # M7: clear any degradation / tap-dead warnings the final flush may have
            # set while stopping. A stopped session's warnings are stale — the actions
            # they suggest ("restart recording") no longer apply.
            with self._warnings_lock:
                self._warnings.clear()
            self._run_cleanup_step("audio file close", self._close_audio_file)

        # Export transcript (reads segments under the transcriber lock).
        self._run_cleanup_step("transcript export", self._export)

        # Mark session completed/errored in DB before the LLM title refinement —
        # otherwise the sidebar keeps showing an ACTIVE badge for the full
        # duration of the (unbounded) title-generation call.
        if self.db and self.db_session_id:
            status = "error" if self.error else "completed"
            self._run_cleanup_step(
                "db stop_session",
                lambda: self.db.stop_session(self.db_session_id, status=status),
            )
            self._run_cleanup_step(
                "knowledge store export", self._schedule_knowledge_store_export
            )

        # Wait for the preliminary title thread — running two mlx-lm
        # generations concurrently crashes Metal. Only refine if it actually
        # finished; a still-alive generation must not have a second one started
        # on top of it.
        # Keep _refine_title synchronous: running it on a background thread
        # races with the screen-capture read thread's Metal cleanup and
        # crashes the process (observed: IOGPUMetalCommandBuffer assertion).
        title_thread_finished = True
        if self._title_thread and self._title_thread.is_alive():
            self._title_thread.join(timeout=TITLE_JOIN_TIMEOUT_S)
            title_thread_finished = not self._title_thread.is_alive()

        if title_thread_finished:
            self._run_cleanup_step("title refinement", self._refine_title)
        else:
            logger.warning(
                "Preliminary title thread still alive after %.0fs; skipping refine "
                "pass to avoid a concurrent mlx-lm generation",
                TITLE_JOIN_TIMEOUT_S,
            )

        logger.info("Session stopped: %s", self.session_id)

    def _process_join_timeout(self) -> float:
        """How long stop() gives the audio-processing thread to drain.

        Scaled to the configured chunk duration, because that is what sets how
        much audio can still be waiting to be transcribed when stop() is called.
        """
        chunk_duration = self.config.streaming.chunk_duration
        return max(
            PROCESS_JOIN_MIN_TIMEOUT_S, chunk_duration * PROCESS_JOIN_CHUNK_FACTOR
        )

    def _join_process_thread(self) -> bool:
        """Wait out the audio-processing thread's final flush.

        Draining the tail of a long recording is slow, not stuck: the final
        flush transcribes the remaining buffer, which on a backlogged
        transcriber takes many times the chunk duration. Returns False only if
        the thread outlives that whole budget, which stop() reads as stuck.
        """
        thread = self._process_thread
        if thread is None:
            return True

        timeout = self._process_join_timeout()
        thread.join(timeout=timeout)
        if not thread.is_alive():
            return True

        logger.error(
            "Audio-processing thread still alive after %.0fs; skipping final "
            "flush and WAV close to avoid concurrent access",
            timeout,
        )
        return False

    def _close_mic_stream_safely(self, timeout: float = 3.0):
        """Stop/close the mic stream off-thread with a timeout.

        PortAudio's AudioOutputUnitStop can deadlock on a CoreAudio HAL mutex
        (observed with PaMacCore err=-50). We leak the stream rather than freeze.
        """
        stream = self._mic_stream
        self._mic_stream = None
        done = threading.Event()

        def _close():
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("Mic stream close failed")
            finally:
                done.set()

        threading.Thread(target=_close, daemon=True).start()
        if not done.wait(timeout=timeout):
            logger.warning(
                "Mic stream stop timed out after %.1fs — leaking PortAudio stream",
                timeout,
            )

    def _start_mic_capture(self, mix_mode: bool = False):
        """Start capturing audio from the microphone via sounddevice.

        Tries the system default input first, then any other input device. A
        Bluetooth headset that a call app already holds exclusively fails to
        open at every sample rate, and the built-in mic sitting right there is
        a better recording than no recording at all.
        """
        import sounddevice as sd

        callback_fn = self._on_mic_audio if mix_mode else self._on_audio_data

        first_error: Exception | None = None
        avoid_shared = mix_mode and self.config.audio.keep_bluetooth_playback
        shared_name = (
            self._default_input_shares_output_device() if avoid_shared else None
        )
        for device in self._candidate_mic_devices(demote_default=bool(shared_name)):
            try:
                self._open_mic_stream(device, callback_fn, mix_mode)
                self._announce_shared_output_demotion(shared_name)
                return
            except Exception as e:
                first_error = first_error or e
                if device is None:
                    name = "default"
                else:
                    try:
                        name = sd.query_devices(device)["name"]
                    except Exception:
                        name = str(device)
                logger.warning("Mic device %s unavailable: %s", name, e)

        raise first_error or RuntimeError("No input device available")

    def _announce_shared_output_demotion(self, shared_name: str | None) -> None:
        """Report the demotion only once a device is actually open.

        T11: this used to be announced while building the candidate list, before
        anything had been opened. When every built-in candidate failed to open,
        capture fell through to the shared device and the banner still claimed
        the built-in mic was recording — and `_warn_if_mic_is_a_fallback`
        suppressed the correcting message precisely because that banner existed.
        The device that opened is the only thing worth reporting.
        """
        if not shared_name:
            return
        if self._mic_device_name == shared_name:
            logger.warning(
                "Wanted to avoid the shared device %r but every alternative "
                "failed to open — recording it after all; system audio may go "
                "silent if the Bluetooth profile switches", shared_name,
            )
            return
        logger.info(
            "Default mic %r is also the output device — recording %r to keep "
            "playback in A2DP", shared_name, self._mic_device_name,
        )
        self._set_warning(
            "mic_shared_output",
            f"{shared_name} is in use for playback — recording with "
            f"{self._mic_device_name} instead so system audio stays in stereo.",
        )

    def _warn_if_mic_is_a_fallback(self) -> None:
        """Say so when the recording is using a mic the user did not pick."""
        import sounddevice as sd

        if not self._mic_device_name:
            return
        try:
            default_name = sd.query_devices(kind="input")["name"]
        except Exception:
            return
        if default_name == self._mic_device_name:
            return
        with self._warnings_lock:
            deliberate = "mic_shared_output" in self._warnings
        if deliberate:
            return
        logger.warning(
            "Default mic %r unavailable; recording with %r instead",
            default_name, self._mic_device_name,
        )
        self._set_warning(
            "mic_fallback",
            f"{default_name} was unavailable — recording with "
            f"{self._mic_device_name} instead.",
        )

    def _on_tap_dead(self) -> None:
        """Called by screen_capture when the Swift tap emits '[tap] dead'."""
        if not self.is_active:
            return
        self._tap_dead = True
        self._set_warning(
            "tap_dead",
            "System audio capture has stopped — the audio chain failed repeatedly. "
            "Try stopping and restarting the recording.",
        )

    def _candidate_mic_devices(
        self, demote_default: bool = False
    ) -> list[int | None]:
        """The default input, then the built-in mic, then anything else.

        The built-in mic outranks the rest of the fallbacks because the other
        candidates are typically Continuity devices — an iPhone that may be
        face-down in a pocket makes a worse recording than the Mac's own mic.

        ``demote_default`` moves the default input below the built-in mic. The
        caller decides that (see ``_default_input_shares_output_device``); this
        method only orders candidates and never touches session state. macOS has
        one bidirectional Bluetooth profile, so opening an AirPods microphone
        drops the link to HFP — mono, 24 kHz — and silences a system-audio tap
        built against the A2DP format. Recording the Mac's own mic costs some
        proximity and keeps playback in A2DP.
        """
        import sounddevice as sd

        candidates: list[int | None] = [None]
        try:
            default_index = sd.default.device[0]
            # W10: only fall back to built-in mics. Continuity (iPhone) and
            # remote devices make worse recordings than the machine's own mic
            # and should not be auto-selected without explicit user intent.
            built_ins = [
                index
                for index, device in enumerate(sd.query_devices())
                if device["max_input_channels"] > 0
                and index != default_index
                and _is_built_in_mic(device["name"])
            ]
            if demote_default and built_ins:
                # Default last, not never: a mono recording still beats none.
                candidates = [*built_ins, None]
            else:
                candidates.extend(built_ins)
        except Exception:
            logger.debug("Could not enumerate input devices", exc_info=True)
        return candidates

    def _default_input_shares_output_device(self) -> str | None:
        """The device name when the default input is also the default output.

        Name equality is the available signal: CoreAudio lists a bidirectional
        Bluetooth headset as one input and one output entry under the same
        name, and no public API exposes the transport they share.
        """
        import sounddevice as sd

        try:
            input_name = sd.query_devices(kind="input")["name"]
            output_name = sd.query_devices(kind="output")["name"]
        except Exception:
            logger.debug("Could not query the default devices", exc_info=True)
            return None
        if input_name and input_name == output_name:
            return input_name
        return None

    def _open_mic_stream(self, device, callback_fn, mix_mode: bool) -> None:
        """Open and start one input device, resampling to the target rate."""
        import numpy as np
        import sounddevice as sd

        target_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels

        # Try the target rate first; fall back to the device's default rate
        try:
            device_info = sd.query_devices(device, kind="input")
            device_rate = int(device_info["default_samplerate"])
        except Exception:
            device_rate = target_rate

        try:
            sd.check_input_settings(
                device=device, samplerate=target_rate, channels=channels
            )
            actual_rate = target_rate
        except Exception:
            logger.info("Mic doesn't support %dHz, using native %dHz with resampling", target_rate, device_rate)
            actual_rate = device_rate

        needs_resample = actual_rate != target_rate

        def mic_callback(indata, frames, time_info, status):
            if status:
                logger.debug("Mic status: %s", status)
            samples = indata[:, 0]
            if needs_resample:
                # Resample from actual_rate to target_rate using linear interpolation
                n_target = int(len(samples) * target_rate / actual_rate)
                indices = np.linspace(0, len(samples) - 1, n_target)
                samples = np.interp(indices, np.arange(len(samples)), samples)
            pcm = (samples * 32767).astype(np.int16).tobytes()
            callback_fn(pcm)

        stream = sd.InputStream(
            device=device,
            samplerate=actual_rate,
            channels=channels,
            dtype="float32",
            callback=mic_callback,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self._mic_stream = stream
        self._mic_device_name = sd.query_devices(device, kind="input")["name"]
        logger.info(
            "Started microphone capture (device=%s, device_rate=%s, target_rate=%s, resample=%s, mix_mode=%s)",
            self._mic_device_name, actual_rate, target_rate, needs_resample, mix_mode,
        )

    def _chunk_pcm_byte_size(self) -> int:
        """Bytes in one streaming chunk of raw PCM int16 audio."""
        sample_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels
        chunk_duration = self.config.streaming.chunk_duration
        return int(sample_rate * channels * 2 * chunk_duration)

    def _audio_buffer_cap_bytes(self) -> int:
        """Maximum bytes retained in the live PCM buffer before dropping oldest audio."""
        return self._chunk_pcm_byte_size() * AUDIO_BUFFER_CAP_FACTOR

    def _append_pcm_with_backpressure(self, buffer: bytearray, data: bytes) -> None:
        """Append PCM to a live buffer, dropping oldest frame-aligned bytes if over the cap."""
        cap = self._audio_buffer_cap_bytes()
        frame_bytes = self._pcm_frame_bytes()
        overflow = len(buffer) + len(data) - cap
        if overflow > 0:
            drop = min(
                len(buffer),
                ((overflow + frame_bytes - 1) // frame_bytes) * frame_bytes,
            )
            if drop > 0:
                del buffer[:drop]
                now = time.monotonic()
                if (
                    now - self._last_buffer_overflow_log
                    >= AUDIO_BUFFER_OVERFLOW_LOG_INTERVAL_S
                ):
                    logger.warning(
                        "Audio buffer exceeded cap (%d bytes); dropped %d oldest bytes "
                        "because transcription is behind",
                        cap,
                        drop,
                    )
                    self._last_buffer_overflow_log = now
        buffer.extend(data)

    def _on_audio_data(self, data: bytes):
        with self._buffer_lock:
            self._append_pcm_with_backpressure(self._audio_buffer, data)

    def _on_system_audio(self, data: bytes):
        with self._buffer_lock:
            self._append_pcm_with_backpressure(self._system_buffer, data)

    def _on_mic_audio(self, data: bytes):
        with self._buffer_lock:
            self._append_pcm_with_backpressure(self._mic_buffer, data)

    def _pcm_frame_bytes(self) -> int:
        """Bytes per PCM frame (one sample across all channels, int16)."""
        return max(self.config.audio.channels * 2, 1)

    def _align_dual_buffers_to_trailing_window(self) -> None:
        """Drop oldest PCM from the longer buffer so system/mic share one trailing window."""
        sys_len = len(self._system_buffer)
        mic_len = len(self._mic_buffer)
        if sys_len == 0 or mic_len == 0 or sys_len == mic_len:
            return

        frame_bytes = self._pcm_frame_bytes()
        if sys_len > mic_len:
            longer = self._system_buffer
            target_len = mic_len
        else:
            longer = self._mic_buffer
            target_len = sys_len

        excess = len(longer) - target_len
        drop = (excess // frame_bytes) * frame_bytes
        if drop == 0:
            drop = excess
        del longer[:drop]
        remaining = len(longer) - target_len
        if remaining > 0:
            del longer[:remaining]

    def _mix_buffers(self) -> bytes:
        """Mix system and mic PCM buffers into one. Must be called under _buffer_lock."""
        import numpy as np

        self._align_dual_buffers_to_trailing_window()

        sys_bytes = bytes(self._system_buffer)
        mic_bytes = bytes(self._mic_buffer)

        if not sys_bytes and not mic_bytes:
            return b""
        if not sys_bytes:
            return mic_bytes
        if not mic_bytes:
            return sys_bytes

        sys_samples = np.frombuffer(sys_bytes, dtype=np.int16).astype(np.float32)
        mic_samples = np.frombuffer(mic_bytes, dtype=np.int16).astype(np.float32)

        mic_boost = self.config.audio.mic_boost
        mixed = sys_samples + mic_samples * mic_boost
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
        return mixed.tobytes()

    def _track_system_silence(self, system_pcm: bytes) -> None:
        """Warn (and keep warning) when the system tap delivers effectively-silent audio.

        B4: absent source (empty bytes) advances wall-clock silence so a tap
        that stops delivering bytes at all also trips the warning. Uses
        max(frame-based, wall-clock) to preserve existing test behavior.
        """
        import numpy as np

        sample_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels
        now = time.monotonic()

        if self._system_tracking_started == 0.0:
            self._system_tracking_started = now

        if system_pcm:
            seconds = len(system_pcm) / max(sample_rate * channels * 2, 1)
            samples = np.frombuffer(system_pcm, dtype=np.int16)
            peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0

            if peak >= SYSTEM_SIGNAL_PEAK_THRESHOLD_INT16:
                if self._system_degraded:
                    self._system_degraded = False
                    self._clear_warning("system")
                # W4: real signal clears the tap_dead latch (tap recovered).
                if self._tap_dead:
                    self._tap_dead = False
                    self._clear_warning("tap_dead")
                self._silent_system_seconds = 0.0
                self._last_system_signal_ts = now
                return

            self._silent_system_seconds += seconds

        # B4: wall-clock fills the gap when the source delivers no bytes.
        last_active = self._last_system_signal_ts or self._system_tracking_started
        wall_clock_silent = now - last_active
        silent_seconds = max(self._silent_system_seconds, wall_clock_silent)

        # A tap that has produced signal and then stopped is a real fault. One
        # that has never produced any is usually just an idle output device.
        saw_signal = self._last_system_signal_ts > 0.0
        threshold = (
            SYSTEM_SILENCE_WARN_SECONDS if saw_signal
            else SYSTEM_SILENCE_COLD_WARN_SECONDS
        )
        if silent_seconds < threshold:
            return

        if not self._system_degraded:
            self._system_degraded = True
            logger.warning(
                "System audio has been effectively silent for %.0fs — the tap is running "
                "but capturing nothing (check the output device)",
                silent_seconds,
            )
        # W3: mic_active is True only when mic is actually capturing in this session.
        mic_active = (
            self.config.audio.audio_source in ("both", "mic") and self._mic_stream is not None
        )
        minutes = int(threshold // 60)
        if mic_active:
            detail = (
                f"{minutes} min — only your microphone "
                "is being recorded. Check your audio output device."
            )
        else:
            detail = (
                f"{minutes} min — nothing is being recorded. "
                "Check your audio output device."
            )
        self._set_warning("system", f"No system audio captured for {detail}")

    def _track_mic_degradation(self, mic_pcm: bytes) -> None:
        """Warn (and keep warning) when the mic sustains suspiciously low RMS.

        B1: uses RMS (not peak). Real noise at -50 dBFS has RMS ≈ 103 int16
        (below threshold 184) but peaks of 1400–3000 int16 (above threshold),
        so peak comparison always takes the "healthy" branch — wrong.

        B4: absent source (empty bytes) advances wall-clock silence so a mic
        that stops delivering bytes also trips the warning.
        """
        import numpy as np

        sample_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels
        now = time.monotonic()

        if self._mic_tracking_started == 0.0:
            self._mic_tracking_started = now

        if mic_pcm:
            seconds = len(mic_pcm) / max(sample_rate * channels * 2, 1)
            samples = np.frombuffer(mic_pcm, dtype=np.int16)
            rms = (
                float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                if len(samples) > 0
                else 0.0
            )

            if rms >= MIC_LOW_RMS_THRESHOLD_INT16:
                if self._mic_degraded:
                    self._mic_degraded = False
                    self._clear_warning("mic")
                self._mic_rms_low_seconds = 0.0
                self._last_mic_signal_ts = now
                return

            self._mic_rms_low_seconds += seconds

        # B4: wall-clock fills the gap when source delivers no bytes.
        last_active = self._last_mic_signal_ts or self._mic_tracking_started
        wall_clock_silent = now - last_active
        silent_seconds = max(self._mic_rms_low_seconds, wall_clock_silent)

        if silent_seconds < MIC_LOW_RMS_WARN_SECONDS:
            return

        if not self._mic_degraded:
            self._mic_degraded = True
            logger.warning(
                "Microphone level has been below RMS threshold (%d dBFS) for %.0fs "
                "— mic may be degraded or muted",
                _MIC_THRESHOLD_DBFS, silent_seconds,
            )
        self._set_warning(
            "mic",
            "Microphone appears degraded — audio level is very low. "
            "Check your microphone or switch to a different input device.",
        )

    def _process_loop(self):
        chunk_duration = self.config.streaming.chunk_duration

        while not self._stop_event.is_set():
            self._stop_event.wait(chunk_duration)
            self._flush_buffer()

        # Final flush
        self._flush_buffer()

    def _flush_buffer(self):
        """Take accumulated PCM from buffer, build WAV, and transcribe."""
        sample_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels
        # Need at least 0.5s of audio
        min_bytes = int(sample_rate * channels * 2 * 0.5)

        audio_source = self.config.audio.audio_source
        with self._buffer_lock:
            if audio_source == "both":
                system_pcm = bytes(self._system_buffer)
                mic_pcm = bytes(self._mic_buffer)
                pcm_data = self._mix_buffers()
                self._system_buffer = bytearray()
                self._mic_buffer = bytearray()
            else:
                pcm_data = bytes(self._audio_buffer)
                system_pcm = pcm_data if audio_source == "system" else b""
                mic_pcm = pcm_data if audio_source == "mic" else b""
                self._audio_buffer = bytearray()

        # B2: run degradation trackers BEFORE the min_bytes guard so a wedged
        # Swift child that delivers no bytes still advances wall-clock silence
        # detection and eventually trips the warning.
        if audio_source in ("both", "system"):
            self._track_system_silence(system_pcm)
        if audio_source in ("both", "mic") and self._mic_stream:
            self._track_mic_degradation(mic_pcm)

        # W4: re-assert the tap_dead warning each flush while the tap is known
        # dead, so it survives banner-dismiss cycles without needing a new event.
        # M7: guard is_active so the final buffer flush after stop() does not set
        # a stale warning on a session that has already ended.
        if self._tap_dead and self.is_active:
            self._set_warning(
                "tap_dead",
                "System audio capture has stopped — the audio chain failed repeatedly. "
                "Try stopping and restarting the recording.",
            )

        if len(pcm_data) < min_bytes:
            return

        # Tee PCM data to the WAV file for playback
        if self._audio_writer:
            try:
                self._audio_writer.writeframes(pcm_data)
            except Exception as e:
                logger.error("Failed to write audio data: %s", e)

        wav_data = _build_wav(pcm_data, sample_rate, channels)

        if self.transcriber:
            from escriba.transcribe.streaming_mlx import ChunkProcessingError

            try:
                self.transcriber.process_wav_chunk(wav_data)
                self._write_new_segments_to_db()
            except ChunkProcessingError as e:
                # Audio was captured (already teed to the WAV above) but could
                # not be transcribed. The transcriber advances its own clock on
                # failure, so later segments stay in sync; surface the gap
                # distinctly rather than treating it as silence.
                logger.warning(
                    "Chunk not transcribed (audio retained in recording): %s", e
                )
            except Exception as e:
                logger.error("Error transcribing chunk: %s", e, exc_info=True)

    def _write_new_segments_to_db(self):
        """Write any new segments to the database (avoids duplicates)."""
        if not self.db or not self.db_session_id:
            return
        segments = self.get_segments()
        new_segments = segments[self._last_segment_count:]
        if new_segments:
            self.db.add_segments(self.db_session_id, new_segments)
            self._last_segment_count = len(segments)
            self._maybe_generate_title()

    def _maybe_generate_title(self):
        """Trigger preliminary title generation once enough segments exist."""
        if self._title_generated or not self.config.auto_name.enabled:
            return
        segments = self.get_segments()
        if len(segments) < self.config.auto_name.min_segments:
            return
        self._title_generated = True
        self._title_thread = threading.Thread(
            target=self._generate_title_async, daemon=True
        )
        self._title_thread.start()

    def _generate_title_async(self):
        """Background: generate a short title from the first transcript segments."""
        try:
            segments = self.get_segments()
            words = " ".join(s.get("text", "") for s in segments[:20]).split()
            snippet = " ".join(words[: self.config.auto_name.max_snippet_words])
            if not snippet.strip():
                return

            from escriba.summarize.llm_summary import generate_session_title

            title = generate_session_title(
                snippet,
                app_name=self.detected_app,
                model=self.config.streaming.summary_model,
            )
            if title and self.db and self.db_session_id:
                self.db.rename_session(self.db_session_id, title)
                logger.info("Auto-named session: %s", title)
                # Preliminary title succeeded — skip the refined pass on stop.
                # Running two mlx-lm generations per session is (a) redundant
                # for most meetings, (b) makes stop() block for ~10s, and (c)
                # widens the window for MLX-concurrency crashes.
                self._title_refined = True
        except Exception:
            logger.debug("Preliminary title generation failed", exc_info=True)

    def _refine_title(self):
        """Generate a refined title using the full transcript (called on stop)."""
        if self._title_refined or not self.config.auto_name.enabled:
            return
        self._title_refined = True
        transcript = self.get_transcript()
        if not transcript.strip():
            return
        try:
            words = transcript.split()
            snippet = " ".join(words[: self.config.auto_name.max_snippet_words])

            from escriba.summarize.llm_summary import generate_session_title

            title = generate_session_title(
                snippet,
                app_name=self.detected_app,
                model=self.config.streaming.summary_model,
            )
            if title and self.db and self.db_session_id:
                self.db.rename_session(self.db_session_id, title)
                logger.info("Refined session title: %s", title)
        except Exception:
            logger.debug("Refined title generation failed", exc_info=True)

    def get_transcript(self) -> str:
        if self.transcriber:
            return self.transcriber.get_full_transcript()
        return ""

    def get_segments(self) -> list[dict[str, Any]]:
        if not self.transcriber:
            return []
        with self.transcriber.lock:
            return list(self.transcriber.segments)

    def consume_error(self) -> str | None:
        """Return the pending error once, then forget it.

        A stop-time failure is news exactly once. The session object outlives
        the recording, so a sticky ``error`` would make every later status poll
        re-raise the same banner forever.
        """
        error, self.error = self.error, None
        return error

    def consume_warning(self) -> str | None:
        """Return one pending warning message (by insertion order) and remove it.

        Thin wrapper over consume_warning_item so both callers share the same
        pop logic; four test files assert against this signature, server.py uses
        consume_warning_item for the source key.
        """
        item = self.consume_warning_item()
        return item[1] if item else None

    def consume_warning_item(self) -> tuple[str, str] | None:
        """Return (source_key, message) for one pending warning and remove it.

        W2: callers that need to key dismissal on source (e.g. the server adding
        warning_source to the status response) use this instead of consume_warning.
        """
        with self._warnings_lock:
            if not self._warnings:
                return None
            key = next(iter(self._warnings))
            return key, self._warnings.pop(key)

    def peek_warning_item(self) -> tuple[str, str] | None:
        """Return (source_key, message) for the highest-priority pending warning without removing it.

        W1: used by _get_status while is_active so one-shot warnings (direct,
        mic_fallback) survive status calls that do not render the warning field.
        B1: iterates _WARNING_PRIORITY so a severe warning (tap_dead) inserted after
        a low-severity one (direct) is never starved by insertion order.
        """
        with self._warnings_lock:
            if not self._warnings:
                return None
            for key in _WARNING_PRIORITY:
                if key in self._warnings:
                    return key, self._warnings[key]
            key = next(iter(self._warnings))
            return key, self._warnings[key]

    def get_status(self) -> dict[str, Any]:
        elapsed = ""
        if self.start_time:
            delta = datetime.now() - self.start_time
            minutes, seconds = divmod(int(delta.total_seconds()), 60)
            hours, minutes = divmod(minutes, 60)
            elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return {
            "is_active": self.is_active,
            "session_id": self.session_id,
            "elapsed": elapsed,
            "segments_count": len(self.get_segments()),
            "error": self.error,
        }

    def _export(self):
        if not self.transcriber:
            return
        formats = self.config.streaming.export_formats
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.transcriber.export_transcript(formats, self.output_dir)
        except Exception as e:
            logger.error("Error exporting transcript: %s", e, exc_info=True)

    def generate_notes(
        self, prompt: str | None = None, model: str | None = None
    ) -> str | None:
        transcript = self.get_transcript()
        if not transcript:
            return None

        effective_model = model or self.config.streaming.summary_model

        user_notes = ""
        if self.db and self.db_session_id:
            user_notes = self.db.get_user_notes(self.db_session_id)

        if prompt or user_notes:
            notes = _generate_custom_notes(
                transcript,
                prompt or "",
                effective_model,
                system_prompt=self.config.prompts.effective_system_prompt,
                user_notes=user_notes,
            )
        else:
            from escriba.summarize import generate_summary

            result = generate_summary(transcript, model=effective_model)
            notes = _summary_to_markdown(result) if result else None

        if notes and self.db and self.db_session_id:
            self.db.append_notes(self.db_session_id, notes)
        return notes

    def _schedule_knowledge_store_export(self) -> None:
        """Fire-and-forget knowledge export so stop() is not blocked on I/O."""
        if not self.db or not self.db_session_id:
            return
        from escriba.knowledge.export_worker import run_knowledge_store_export

        session_id = self.db_session_id
        db = self.db
        config = self.config
        audio_file = self._audio_file
        threading.Thread(
            target=run_knowledge_store_export,
            args=(db, session_id, config, audio_file),
            daemon=True,
            name=f"knowledge-export-{session_id[:8]}",
        ).start()

def _markdown_list_item(value: object) -> str | None:
    """Return stripped text for a bullet item, or None when empty."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    return text or None


def _action_item_to_markdown(item: object) -> str | None:
    """Format one action item dict as a markdown bullet line."""
    if isinstance(item, dict):
        task = _markdown_list_item(item.get("task"))
        if not task:
            return None
        assignee = _markdown_list_item(item.get("assignee"))
        due_date = _markdown_list_item(item.get("due_date"))
        line = task
        if assignee:
            line = f"{line} — {assignee}"
        if due_date:
            line = f"{line} (due: {due_date})"
        return line
    return _markdown_list_item(item)


def _summary_to_markdown(result: dict[str, Any]) -> str:
    """Convert a generate_summary payload into dashboard-friendly markdown."""
    sections: list[str] = []

    summary = _markdown_list_item(result.get("summary"))
    if summary:
        sections.append(f"## Summary\n\n{summary}")

    key_points = result.get("key_points")
    if isinstance(key_points, list):
        bullets = [
            f"- {text}"
            for item in key_points
            if (text := _markdown_list_item(item))
        ]
        if bullets:
            sections.append("## Key Points\n\n" + "\n".join(bullets))

    action_items = result.get("action_items")
    if isinstance(action_items, list):
        bullets = [
            f"- {line}"
            for item in action_items
            if (line := _action_item_to_markdown(item))
        ]
        if bullets:
            sections.append("## Action Items\n\n" + "\n".join(bullets))

    decisions = result.get("decisions")
    if isinstance(decisions, list):
        bullets = [
            f"- {text}"
            for item in decisions
            if (text := _markdown_list_item(item))
        ]
        if bullets:
            sections.append("## Decisions\n\n" + "\n".join(bullets))

    topics = result.get("topics")
    if isinstance(topics, list):
        bullets = [
            f"- {text}"
            for item in topics
            if (text := _markdown_list_item(item))
        ]
        if bullets:
            sections.append("## Topics\n\n" + "\n".join(bullets))

    return "\n\n".join(sections)


def _build_wav(pcm_data: bytes, sample_rate: int, channels: int) -> bytes:
    """Build a WAV file from raw PCM int16 data."""
    bits_per_sample = 16
    data_size = len(pcm_data)
    header = b"RIFF"
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)
    header += struct.pack("<H", 1)  # PCM
    header += struct.pack("<H", channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", sample_rate * channels * bits_per_sample // 8)
    header += struct.pack("<H", channels * bits_per_sample // 8)
    header += struct.pack("<H", bits_per_sample)
    header += b"data"
    header += struct.pack("<I", data_size)
    return header + pcm_data


def _build_transcriber(config, *, realtime_output: bool) -> Any:
    """
    Construct the streaming transcriber for the configured backend.

    Shared by live sessions and re-transcribe so both paths honor the same
    VAD, hallucination, and dictionary settings from ``config``.
    """
    model_size = config.streaming.model_size
    language = config.streaming.language
    shared_kwargs = {
        "model_size": model_size,
        "language": language,
        "realtime_output": realtime_output,
        "vad_enabled": config.streaming.vad_enabled,
        "vad_config": config.vad,
        "hallucination_config": config.hallucination,
    }

    if config.streaming.backend == "mlx-whisper":
        try:
            from escriba.transcribe.streaming_mlx import StreamingTranscriberMLX

            return StreamingTranscriberMLX(
                **shared_kwargs,
                dictionary=config.dictionary,
            )
        except ImportError:
            logger.warning(
                "mlx-whisper backend unavailable; falling back to faster-whisper"
            )

    from escriba.transcribe.streaming import StreamingTranscriber

    return StreamingTranscriber(
        **shared_kwargs,
        device=config.streaming.device,
    )


def _build_custom_prompt(
    transcript: str,
    prompt: str,
    system_prompt: str | None = None,
    user_notes: str = "",
) -> str:
    """
    Render the system-prompt template with the transcript and user instruction.

    ``system_prompt`` may contain ``{transcript}``, ``{prompt}``, and
    optionally ``{user_notes}`` placeholders.  Falls back to the built-in
    default if it is empty or malformed.

    When the template has no ``{user_notes}`` placeholder but user_notes is
    non-empty, the notes are prepended as an XML block before the rendered text.
    """
    from escriba.config import DEFAULT_SYSTEM_PROMPT

    template = (system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    has_user_notes_placeholder = "{user_notes}" in template
    fmt_kwargs: dict[str, str] = {
        "transcript": transcript,
        "prompt": prompt,
    }
    if has_user_notes_placeholder:
        fmt_kwargs["user_notes"] = user_notes or ""
    try:
        rendered = template.format(**fmt_kwargs)
    except (KeyError, IndexError, ValueError):
        logger.warning("Invalid system prompt template; using default")
        try:
            rendered = DEFAULT_SYSTEM_PROMPT.format(
                transcript=transcript, prompt=prompt, user_notes=user_notes or ""
            )
        except Exception:
            logger.warning("DEFAULT_SYSTEM_PROMPT format also failed; using bare fallback", exc_info=True)
            rendered = f"{transcript}\n\n{prompt}"

    if not has_user_notes_placeholder and user_notes:
        preamble = f"<user_notes>\n{user_notes}\n</user_notes>\n\n"
        rendered = preamble + rendered

    return rendered


def _generate_custom_notes(
    transcript: str,
    prompt: str,
    model: str = "gemini",
    system_prompt: str | None = None,
    user_notes: str = "",
) -> str | None:
    """Generate notes from transcript with a custom user prompt."""
    from escriba.summarize.llm_summary import (
        DEFAULT_CLAUDE_MODEL,
        DEFAULT_GEMINI_MODEL,
        _call_llm_claude,
        _call_llm_gemini,
        _call_llm_local,
        recommend_model,
        resolve_provider_and_model,
    )

    full_prompt = _build_custom_prompt(transcript, prompt, system_prompt, user_notes=user_notes)

    provider, model_id = resolve_provider_and_model(model)

    if provider in ("local", "gemini", "claude") and not model_id:
        if provider == "local":
            model_id = recommend_model()
        elif provider == "gemini":
            model_id = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        else:
            model_id = os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL

    try:
        if provider == "local":
            if not model_id:
                logger.error("No local model available for notes")
                return None

            # Thinking off: on a long transcript the reasoning block eats the
            # whole token budget and the model never reaches its answer.
            return _call_llm_local(
                full_prompt, model_id, max_tokens=4096, enable_thinking=False
            )
        elif provider == "gemini":
            resolved_model = model_id or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
            return _call_llm_gemini(full_prompt, resolved_model)
        elif provider == "claude":
            resolved_model = model_id or os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL
            return _call_llm_claude(full_prompt, resolved_model)
        else:
            if provider == "none":
                logger.info("No AI provider available — skipping notes")
            else:
                logger.error("Unsupported provider for notes: %s", provider)
            return None
    except Exception as e:
        logger.error("Error generating notes: %s", e, exc_info=True)
        return None


def retranscribe_from_wav(audio_path: Path, config) -> list[dict]:
    """Re-transcribe a WAV file and return segments."""
    import wave as wave_mod

    with wave_mod.open(str(audio_path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        total_frames = wf.getnframes()
        all_pcm = wf.readframes(total_frames)

    backend = config.streaming.backend
    model_size = config.streaming.model_size
    chunk_duration = config.streaming.chunk_duration

    logger.info(
        "Re-transcribing %s: %d frames, backend=%s, model=%s",
        audio_path.name, total_frames, backend, model_size,
    )

    transcriber = _build_transcriber(config, realtime_output=False)

    chunk_bytes = int(sample_rate * channels * 2 * chunk_duration)
    min_bytes = sample_rate * 2  # at least 0.5s

    for offset in range(0, len(all_pcm), chunk_bytes):
        chunk_pcm = all_pcm[offset:offset + chunk_bytes]
        if len(chunk_pcm) < min_bytes:
            continue
        wav_data = _build_wav(chunk_pcm, sample_rate, channels)
        try:
            transcriber.process_wav_chunk(wav_data)
        except Exception as e:
            logger.error("Error in re-transcribe chunk: %s", e)

    segments = list(transcriber.segments)
    logger.info("Re-transcription complete: %d segments", len(segments))
    return segments
