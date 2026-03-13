"""Streaming transcriber using mlx-whisper for Apple Silicon GPU acceleration."""

from __future__ import annotations

import io
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from escriba.transcribe.config import HallucinationConfig, VADConfig
from escriba.transcribe.metrics import CaptureMetrics

logger = logging.getLogger(__name__)

# Verificar si mlx-whisper está disponible
try:
    import mlx_whisper

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mlx_whisper = None

# Map model sizes to HuggingFace repo paths
MLX_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


class StreamingTranscriberMLX:
    """
    Transcripción en tiempo real usando mlx-whisper.

    Optimizado para Apple Silicon con aceleración GPU real.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "es",
        output_file: Optional[Path] = None,
        vad_enabled: bool = True,
        realtime_output: bool = True,
        vad_config: Optional[VADConfig] = None,
        hallucination_config: Optional[HallucinationConfig] = None,
        metrics: Optional[CaptureMetrics] = None,
    ):
        """
        Inicializa el transcriber MLX.

        Args:
            model_size: Tamaño del modelo (tiny, base, small, medium, large, turbo)
            language: Idioma para transcripción (código ISO 639-1)
            output_file: Archivo donde escribir transcripciones en tiempo real
            vad_enabled: Habilitar Voice Activity Detection
            realtime_output: Mostrar transcripciones en consola en tiempo real
            vad_config: Configuración VAD. Si es None, se carga desde variables de entorno.
            hallucination_config: Configuración para prevención de alucinaciones.
            metrics: Instancia de CaptureMetrics para tracking de métricas (opcional)
        """
        if not MLX_AVAILABLE:
            raise ImportError(
                "mlx-whisper not available. Install with: pip install mlx-whisper"
            )

        self.model_size = model_size
        self.language = language
        self.output_file = output_file
        self.vad_enabled = vad_enabled
        self.realtime_output = realtime_output
        self.metrics = metrics

        # Configurar VAD y hallucination prevention
        self.vad_config = vad_config or VADConfig.from_env()
        self.hallucination_config = (
            hallucination_config or HallucinationConfig.from_env()
        )

        # Buffer para mantener contexto entre chunks
        self.transcription_buffer: list[str] = []
        self.segments: list[dict[str, Any]] = []
        self.start_time = time.time()
        self.accumulated_audio_time = 0.0

        # Lock para thread-safety
        self.lock = threading.Lock()

        # Resolve model repo path and cache locally to avoid HF network checks
        repo_id = MLX_MODEL_REPOS.get(
            model_size, f"mlx-community/whisper-{model_size}"
        )
        self.model_path = self._resolve_model_path(repo_id)
        logger.info("Using MLX Whisper model: %s (path: %s)", repo_id, self.model_path)

    @staticmethod
    def _resolve_model_path(repo_id: str) -> str:
        """Resolve HF repo to a local path. Downloads on first use, then uses cache."""
        from huggingface_hub import snapshot_download

        try:
            # Try local cache first (no network)
            return snapshot_download(repo_id, local_files_only=True)
        except Exception:
            # Not cached yet — download once
            logger.info("Downloading model %s (first time only)...", repo_id)
            return snapshot_download(repo_id)

    def process_wav_chunk(self, wav_data: bytes) -> Optional[str]:
        """
        Procesa un chunk de audio en formato WAV.

        Args:
            wav_data: Audio en formato WAV completo (con header)

        Returns:
            Texto transcrito o None si no hay voz detectada
        """
        # Registrar inicio de procesamiento para métricas
        start_timestamp = None
        if self.metrics:
            start_timestamp = self.metrics.record_chunk_start()
            self.metrics.record_audio_level(wav_data)

        try:
            import wave

            if len(wav_data) < 44:
                logger.warning("WAV chunk too small: %s bytes", len(wav_data))
                if self.metrics and start_timestamp:
                    self.metrics.record_chunk_end(
                        start_timestamp, had_transcription=False
                    )
                return None

            if wav_data[:4] != b"RIFF" or wav_data[8:12] != b"WAVE":
                logger.warning("Invalid WAV header")
                if self.metrics and start_timestamp:
                    self.metrics.record_chunk_end(
                        start_timestamp, had_transcription=False
                    )
                return None

            # Decode WAV to numpy array — passes audio directly to mlx-whisper,
            # avoiding the ffmpeg dependency that load_audio() requires.
            import numpy as np

            wav_io = io.BytesIO(wav_data)
            with wave.open(wav_io, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                n_channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                chunk_duration = n_frames / sample_rate if sample_rate > 0 else 0
                raw_frames = wav_file.readframes(n_frames)

            if sampwidth == 2:
                audio_np = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                audio_np = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio_np = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            if n_channels > 1:
                audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)

            if sample_rate != 16000:
                duration_s = len(audio_np) / sample_rate
                target_len = int(duration_s * 16000)
                audio_np = np.interp(
                    np.linspace(0, len(audio_np), target_len, endpoint=False),
                    np.arange(len(audio_np)),
                    audio_np,
                ).astype(np.float32)

            transcribe_kwargs = {
                "path_or_hf_repo": self.model_path,
                "condition_on_previous_text": self.hallucination_config.condition_on_previous_text,
                "no_speech_threshold": self.hallucination_config.no_speech_threshold,
                "compression_ratio_threshold": self.hallucination_config.compression_ratio_threshold,
                "logprob_threshold": self.hallucination_config.logprob_threshold,
                "hallucination_silence_threshold": 2.0,
            }
            if self.language and self.language != "auto":
                transcribe_kwargs["language"] = self.language
            result = mlx_whisper.transcribe(audio_np, **transcribe_kwargs)

            texts = []
            if result and "segments" in result:
                for segment in result["segments"]:
                    text = segment.get("text", "").strip()
                    if text and not self._is_repetitive(text):
                        texts.append(text)
                        start = segment.get("start", 0.0)
                        end = segment.get("end", start)
                        self._handle_transcription(
                            text,
                            self.accumulated_audio_time + start,
                            self.accumulated_audio_time + end,
                        )
                    elif text:
                        logger.debug("Filtered repetitive text: %s...", text[:50])

            if chunk_duration > 0:
                self.accumulated_audio_time += chunk_duration
                if self.metrics:
                    self.metrics.record_audio_duration(chunk_duration)

            result_text = " ".join(texts) if texts else None

            if self.metrics and start_timestamp:
                self.metrics.record_chunk_end(
                    start_timestamp, had_transcription=(result_text is not None)
                )

            return result_text

        except Exception as e:
            logger.error("Error processing WAV chunk: %s", e, exc_info=True)
            if self.metrics:
                self.metrics.record_error()
                if start_timestamp:
                    self.metrics.record_chunk_end(
                        start_timestamp, had_transcription=False
                    )
            return None

    def _is_repetitive(self, text: str) -> bool:
        """
        Check if text is repetitive (hallucination pattern).

        Returns:
            True if text appears to be repetitive hallucination
        """
        import re

        # Filter out punctuation-only segments (silence hallucinations like ".")
        stripped = text.strip()
        if not stripped or stripped in {".", "..", "...", ",", "!", "?", "-"}:
            return True

        # Filter if just whitespace and punctuation
        if re.match(r"^[\s\.\,\!\?\-\:\;]+$", stripped):
            return True

        words = text.lower().split()

        # For short texts (< 4 words), check if all words are the same
        if len(words) < 4:
            if len(words) >= 2 and len(set(words)) == 1:
                return True  # e.g., "best best" or "los los los"
            return False

        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)

        # If very few unique words relative to total, it's repetitive
        # e.g., "best best best best best best" has ratio 1/6 = 0.17
        if unique_ratio < 0.35:
            return True

        # Check for repeating n-grams (phrases)
        # e.g., "y eso era todo y eso era todo y eso era todo"
        for n in [2, 3, 4, 5]:  # Check 2-5 word phrases
            if len(words) >= n * 2:  # Need at least 2 repetitions
                ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
                ngram_counts = {}
                for ng in ngrams:
                    ngram_counts[ng] = ngram_counts.get(ng, 0) + 1

                # If any n-gram appears 3+ times, it's likely repetitive
                max_count = max(ngram_counts.values()) if ngram_counts else 0
                if max_count >= 3:
                    return True

        return False

    def _handle_transcription(
        self, text: str, start_time: float, end_time: float
    ) -> None:
        """Maneja una transcripción parcial."""
        with self.lock:
            self.transcription_buffer.append(text)
            self.segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                }
            )

            absolute_start = self.start_time + start_time
            timestamp = time.strftime("%H:%M:%S", time.localtime(absolute_start))

            logger.info("Transcription: [%s] %s", timestamp, text)
            if self.realtime_output:
                print(f"[{timestamp}] {text}", flush=True)

            if self.output_file:
                self._append_to_file(
                    text, absolute_start, absolute_start + (end_time - start_time)
                )

    def _append_to_file(self, text: str, start_time: float, end_time: float) -> None:
        """Escribe transcripción a archivo en tiempo real."""
        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%H:%M:%S", time.localtime(start_time))
            with self.output_file.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {text}\n")
                f.flush()
        except Exception as e:
            logger.error("Error writing to file: %s", e, exc_info=True)

    def get_full_transcript(self) -> str:
        """Obtiene la transcripción completa hasta el momento."""
        with self.lock:
            return " ".join(self.transcription_buffer)

    def export_transcript(self, formats: list[str], output_dir: Path) -> None:
        """Exporta la transcripción en los formatos especificados."""
        from escriba.transcribe.formats import (
            export_to_json,
            export_to_txt,
            export_to_srt,
            export_to_markdown,
        )

        with self.lock:
            segments = self.segments.copy()

        if not segments:
            logger.warning("No segments to export")
            return

        # Metadata para JSON
        metadata = {
            "model": self.model_size,
            "language": self.language,
            "device": "mlx",
            "compute_type": "float16",
        }

        base_name = datetime.now().strftime("transcript_%Y%m%d_%H%M%S")

        for fmt in formats:
            if fmt == "json":
                output_path = output_dir / f"{base_name}.json"
                export_to_json(segments, output_path, metadata)
            elif fmt == "txt":
                output_path = output_dir / f"{base_name}.txt"
                export_to_txt(segments, output_path)
            elif fmt == "srt":
                output_path = output_dir / f"{base_name}.srt"
                export_to_srt(segments, output_path)
            elif fmt == "markdown":
                output_path = output_dir / f"{base_name}.md"
                export_to_markdown(segments, output_path)
