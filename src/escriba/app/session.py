"""Transcription session management for the menu bar app."""

from __future__ import annotations

import logging
import os
import struct
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cap live PCM buffer at this multiple of one transcription chunk.
AUDIO_BUFFER_CAP_FACTOR = 2
AUDIO_BUFFER_OVERFLOW_LOG_INTERVAL_S = 5.0


class TranscriptionSession:
    """Manages a single transcription session: capture + transcribe + notes."""

    def __init__(self, config, database=None):
        from escriba.config import AppConfig

        self.config: AppConfig = config
        self.db = database
        self.transcriber = None
        self.screen_capture = None
        self._mic_stream = None
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
        self._audio_file: Path | None = None
        self._audio_writer: wave.Wave_write | None = None
        self.detected_app: str | None = None
        self._title_generated: bool = False
        self._title_refined: bool = False
        self._title_thread: threading.Thread | None = None
        self._last_buffer_overflow_log: float = 0.0

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
        self._last_buffer_overflow_log = 0.0

        # Create DB session
        if self.db:
            backend = self.config.streaming.backend
            model_size = self.config.streaming.model_size
            language = self.config.streaming.language
            name = f"Session {self.start_time.strftime('%Y-%m-%d %H:%M')}"
            self.db_session_id = self.db.create_session(
                name=name, model=model_size, language=language, backend=backend,
            )

        # Open WAV file for audio recording
        self._open_audio_file()

        # Create transcriber based on backend
        backend = self.config.streaming.backend
        model_size = self.config.streaming.model_size
        language = self.config.streaming.language
        logger.info(
            "Session config: backend=%s, model=%s, language=%s",
            backend, model_size, language,
        )

        try:
            if backend == "mlx-whisper":
                from escriba.transcribe.streaming_mlx import (
                    StreamingTranscriberMLX,
                )

                self.transcriber = StreamingTranscriberMLX(
                    model_size=model_size,
                    language=language,
                    realtime_output=True,
                    dictionary=self.config.dictionary,
                )
            else:
                from escriba.transcribe.streaming import StreamingTranscriber

                self.transcriber = StreamingTranscriber(
                    model_size=model_size,
                    language=language,
                    device=self.config.streaming.device,
                    realtime_output=True,
                )
        except Exception as e:
            self.error = f"Failed to load model: {e}"
            logger.error("Failed to load model: %s", e, exc_info=True)
            return

        # Start audio capture based on audio_source mode
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
                )
                if not self.screen_capture.start():
                    self.error = "Failed to start audio capture. Check permissions."
                    return
            except ImportError:
                self.error = (
                    "Swift audio-capture CLI not available. "
                    "Build with: cd swift-audio-capture && swift build -c release"
                )
                logger.error(self.error)
                return

        # Start mic capture (for "mic" and "both" modes)
        if audio_source in ("mic", "both"):
            try:
                self._start_mic_capture(mix_mode=audio_source == "both")
            except Exception as e:
                self.error = f"Failed to start microphone capture: {e}"
                logger.error(self.error, exc_info=True)
                return

        # Start processing thread
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._process_thread.start()
        self.is_active = True
        logger.info("Session started: %s", self.session_id)

    def stop(self):
        if not self.is_active:
            return

        self.is_active = False
        self._stop_event.set()

        if self._mic_stream is not None:
            self._close_mic_stream_safely()

        if self.screen_capture:
            self.screen_capture.stop()

        if self._process_thread:
            self._process_thread.join(timeout=10)

        # Process any remaining audio
        self._flush_buffer()

        # Close audio file and store path
        self._close_audio_file()

        # Export transcript
        self._export()

        # Mark session completed in DB before the LLM title refinement —
        # otherwise the sidebar keeps showing an ACTIVE badge for the full
        # duration of the (unbounded) title-generation call.
        # Keep _refine_title synchronous: running it on a background thread
        # races with the screen-capture read thread's Metal cleanup and
        # crashes the process (observed: IOGPUMetalCommandBuffer assertion).
        if self.db and self.db_session_id:
            status = "error" if self.error else "completed"
            self.db.stop_session(self.db_session_id, status=status)

        # Wait for the preliminary title thread — running two mlx-lm
        # generations concurrently crashes Metal.
        if self._title_thread and self._title_thread.is_alive():
            self._title_thread.join(timeout=30)

        self._refine_title()

        logger.info("Session stopped: %s", self.session_id)

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
        """Start capturing audio from the microphone via sounddevice."""
        import numpy as np
        import sounddevice as sd

        target_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels
        callback_fn = self._on_mic_audio if mix_mode else self._on_audio_data

        # Try the target rate first; fall back to the device's default rate
        try:
            device_info = sd.query_devices(kind="input")
            device_rate = int(device_info["default_samplerate"])
        except Exception:
            device_rate = target_rate

        try:
            sd.check_input_settings(samplerate=target_rate, channels=channels)
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

        self._mic_stream = sd.InputStream(
            samplerate=actual_rate,
            channels=channels,
            dtype="float32",
            callback=mic_callback,
        )
        self._mic_stream.start()
        logger.info(
            "Started microphone capture (device_rate=%s, target_rate=%s, resample=%s, mix_mode=%s)",
            actual_rate, target_rate, needs_resample, mix_mode,
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
        """Append PCM to a live buffer, dropping oldest bytes if over the cap."""
        cap = self._audio_buffer_cap_bytes()
        overflow = len(buffer) + len(data) - cap
        if overflow > 0:
            del buffer[:overflow]
            now = datetime.now().timestamp()
            if now - self._last_buffer_overflow_log >= AUDIO_BUFFER_OVERFLOW_LOG_INTERVAL_S:
                logger.warning(
                    "Audio buffer exceeded cap (%d bytes); dropped %d oldest bytes "
                    "because transcription is behind",
                    cap,
                    overflow,
                )
                self._last_buffer_overflow_log = now
        buffer.extend(data)

    def _on_audio_data(self, data: bytes):
        with self._buffer_lock:
            self._append_pcm_with_backpressure(self._audio_buffer, data)

    def _on_system_audio(self, data: bytes):
        with self._buffer_lock:
            self._system_buffer.extend(data)

    def _on_mic_audio(self, data: bytes):
        with self._buffer_lock:
            self._mic_buffer.extend(data)

    def _mix_buffers(self) -> bytes:
        """Mix system and mic PCM buffers into one. Must be called under _buffer_lock."""
        import numpy as np

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

        # Pad shorter buffer with silence
        max_len = max(len(sys_samples), len(mic_samples))
        if len(sys_samples) < max_len:
            sys_samples = np.pad(sys_samples, (0, max_len - len(sys_samples)))
        if len(mic_samples) < max_len:
            mic_samples = np.pad(mic_samples, (0, max_len - len(mic_samples)))

        mic_boost = self.config.audio.mic_boost
        mixed = sys_samples + mic_samples * mic_boost
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
        return mixed.tobytes()

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

        with self._buffer_lock:
            if self.config.audio.audio_source == "both":
                pcm_data = self._mix_buffers()
                self._system_buffer = bytearray()
                self._mic_buffer = bytearray()
            else:
                pcm_data = bytes(self._audio_buffer)
                self._audio_buffer = bytearray()

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
            try:
                self.transcriber.process_wav_chunk(wav_data)
                self._write_new_segments_to_db()
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

        if prompt:
            return _generate_custom_notes(
                transcript,
                prompt,
                effective_model,
                system_prompt=self.config.prompts.effective_system_prompt,
            )

        from escriba.summarize import generate_summary

        result = generate_summary(transcript, model=effective_model)
        if result:
            return _summary_to_markdown(result)
        return None


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


def _build_custom_prompt(
    transcript: str, prompt: str, system_prompt: str | None = None
) -> str:
    """
    Render the system-prompt template with the transcript and user instruction.

    ``system_prompt`` may contain ``{transcript}`` and ``{prompt}`` placeholders.
    Falls back to the built-in default if it is empty or malformed.
    """
    from escriba.config import DEFAULT_SYSTEM_PROMPT

    template = (system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    try:
        return template.format(transcript=transcript, prompt=prompt)
    except (KeyError, IndexError, ValueError):
        logger.warning("Invalid system prompt template; using default")
        return DEFAULT_SYSTEM_PROMPT.format(transcript=transcript, prompt=prompt)


def _generate_custom_notes(
    transcript: str,
    prompt: str,
    model: str = "gemini",
    system_prompt: str | None = None,
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

    full_prompt = _build_custom_prompt(transcript, prompt, system_prompt)

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

            return _call_llm_local(full_prompt, model_id, max_tokens=4096)
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
    language = config.streaming.language
    chunk_duration = config.streaming.chunk_duration

    logger.info(
        "Re-transcribing %s: %d frames, backend=%s, model=%s",
        audio_path.name, total_frames, backend, model_size,
    )

    if backend == "mlx-whisper":
        from escriba.transcribe.streaming_mlx import StreamingTranscriberMLX

        transcriber: StreamingTranscriberMLX | Any
        transcriber = StreamingTranscriberMLX(
            model_size=model_size,
            language=language,
            realtime_output=False,
            vad_enabled=config.streaming.vad_enabled,
            vad_config=config.vad,
            hallucination_config=config.hallucination,
            dictionary=config.dictionary,
        )
    else:
        from escriba.transcribe.streaming import StreamingTranscriber

        transcriber = StreamingTranscriber(
            model_size=model_size,
            language=language,
            device=config.streaming.device,
            realtime_output=False,
            vad_enabled=config.streaming.vad_enabled,
            vad_config=config.vad,
            hallucination_config=config.hallucination,
        )

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
