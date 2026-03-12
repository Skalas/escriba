"""Transcription session management for the menu bar app."""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TranscriptionSession:
    """Manages a single transcription session: capture + transcribe + notes."""

    def __init__(self, config, database=None):
        from local_transcriber.config import AppConfig

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

        # Create DB session
        if self.db:
            backend = self.config.streaming.backend
            model_size = self.config.streaming.model_size
            language = self.config.streaming.language
            name = f"Session {self.start_time.strftime('%Y-%m-%d %H:%M')}"
            self.db_session_id = self.db.create_session(
                name=name, model=model_size, language=language, backend=backend,
            )

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
                from local_transcriber.transcribe.streaming_mlx import (
                    StreamingTranscriberMLX,
                )

                self.transcriber = StreamingTranscriberMLX(
                    model_size=model_size,
                    language=language,
                    realtime_output=True,
                )
            else:
                from local_transcriber.transcribe.streaming import StreamingTranscriber

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
                from local_transcriber.audio.screen_capture import (
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
            self._mic_stream.stop()
            self._mic_stream.close()
            self._mic_stream = None

        if self.screen_capture:
            self.screen_capture.stop()

        if self._process_thread:
            self._process_thread.join(timeout=10)

        # Process any remaining audio
        self._flush_buffer()

        # Export transcript
        self._export()

        # Update DB
        if self.db and self.db_session_id:
            status = "error" if self.error else "completed"
            self.db.stop_session(self.db_session_id, status=status)

        logger.info("Session stopped: %s", self.session_id)

    def _start_mic_capture(self, mix_mode: bool = False):
        """Start capturing audio from the microphone via sounddevice."""
        import numpy as np
        import sounddevice as sd

        sample_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels
        callback_fn = self._on_mic_audio if mix_mode else self._on_audio_data

        def mic_callback(indata, frames, time_info, status):
            if status:
                logger.debug("Mic status: %s", status)
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            callback_fn(pcm)

        self._mic_stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=mic_callback,
        )
        self._mic_stream.start()
        logger.info("Started microphone capture (sample_rate=%s, mix_mode=%s)", sample_rate, mix_mode)

    def _on_audio_data(self, data: bytes):
        with self._buffer_lock:
            self._audio_buffer.extend(data)

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
        sample_rate = self.config.audio.sample_rate

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
            return _generate_custom_notes(transcript, prompt, effective_model)

        from local_transcriber.summarize import generate_summary

        result = generate_summary(transcript, model=effective_model)
        if result:
            return json.dumps(result, indent=2, ensure_ascii=False)
        return None


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


def _generate_custom_notes(
    transcript: str, prompt: str, model: str = "gemini"
) -> str | None:
    """Generate notes from transcript with a custom user prompt."""
    full_prompt = (
        "Here is a transcript from a meeting/call:\n\n"
        f"{transcript}\n\n"
        "Based on the above transcript, please do the following:\n"
        f"{prompt}\n\n"
        "Respond in a clear, well-structured format."
    )

    try:
        if model == "gemini":
            return _call_gemini(full_prompt)
        elif model == "claude":
            return _call_claude(full_prompt)
        else:
            logger.error("Unsupported model for notes: %s", model)
            return None
    except Exception as e:
        logger.error("Error generating notes: %s", e, exc_info=True)
        return None


def _call_gemini(prompt: str) -> str | None:
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not set"
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview")
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


def _call_claude(prompt: str) -> str | None:
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set"
    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
