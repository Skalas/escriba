"""Streaming transcriber using mlx-whisper for Apple Silicon GPU acceleration."""

from __future__ import annotations

import io
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from local_transcriber.transcribe.config import VADConfig
from local_transcriber.transcribe.metrics import CaptureMetrics
from local_transcriber.utils.env import get_bool_env, get_float_env

logger = logging.getLogger(__name__)

# Verificar si mlx-whisper está disponible
try:
    import mlx_whisper

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mlx_whisper = None


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
        metrics: Optional[CaptureMetrics] = None,
    ):
        """
        Inicializa el transcriber MLX.

        Args:
            model_size: Tamaño del modelo (tiny, base, small, medium, large)
            language: Idioma para transcripción (código ISO 639-1)
            output_file: Archivo donde escribir transcripciones en tiempo real
            vad_enabled: Habilitar Voice Activity Detection
            realtime_output: Mostrar transcripciones en consola en tiempo real
            vad_config: Configuración VAD. Si es None, se carga desde variables de entorno.
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

        # Configurar VAD
        self.vad_config = vad_config or VADConfig.from_env()

        # Buffer para mantener contexto entre chunks
        self.transcription_buffer: list[str] = []
        self.segments: list[dict[str, Any]] = []
        self.start_time = time.time()
        self.accumulated_audio_time = 0.0

        # Lock para thread-safety
        self.lock = threading.Lock()

        # Cargar modelo MLX
        logger.info(f"Loading MLX Whisper model: {model_size}")
        try:
            # mlx-whisper usa una API diferente
            # Por ahora, cargamos el modelo básico
            # Nota: La API exacta puede variar según la versión de mlx-whisper
            self.model = mlx_whisper.load_model(model_size)
            logger.info("MLX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}", exc_info=True)
            raise

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

            wav_io = io.BytesIO(wav_data)

            if len(wav_data) < 44:
                logger.warning(f"WAV chunk too small: {len(wav_data)} bytes")
                if self.metrics and start_timestamp:
                    self.metrics.record_chunk_end(start_timestamp, had_transcription=False)
                return None

            if wav_data[:4] != b"RIFF" or wav_data[8:12] != b"WAVE":
                logger.warning("Invalid WAV header")
                if self.metrics and start_timestamp:
                    self.metrics.record_chunk_end(start_timestamp, had_transcription=False)
                return None

            with wave.open(wav_io, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                frames = wav_file.readframes(wav_file.getnframes())

                if len(frames) == 0:
                    if self.metrics and start_timestamp:
                        self.metrics.record_chunk_end(start_timestamp, had_transcription=False)
                    return None

                # Convertir a numpy array
                audio_array = np.frombuffer(frames, dtype=np.int16)

                # Convertir a mono si es estéreo
                if n_channels == 2:
                    audio_array = (
                        audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    )

                # Convertir a float32 normalizado
                audio_float = audio_array.astype(np.float32) / 32768.0

                # Transcribir con mlx-whisper
                # Nota: La API de mlx-whisper puede ser diferente
                # Esta es una implementación básica que puede necesitar ajustes
                try:
                    # mlx-whisper típicamente usa transcribe() que retorna segmentos
                    result = mlx_whisper.transcribe(
                        self.model,
                        audio_float,
                        language=self.language if self.language != "auto" else None,
                    )

                    # Procesar resultado
                    texts = []
                    if result and "segments" in result:
                        for segment in result["segments"]:
                            text = segment.get("text", "").strip()
                            if text:
                                texts.append(text)
                                start = segment.get("start", 0.0)
                                end = segment.get("end", start)
                                self._handle_transcription(
                                    text,
                                    self.accumulated_audio_time + start,
                                    self.accumulated_audio_time + end,
                                )

                    # Actualizar tiempo acumulado
                    if len(audio_float) > 0:
                        chunk_duration = len(audio_float) / sample_rate
                        self.accumulated_audio_time += chunk_duration
                        if self.metrics:
                            self.metrics.record_audio_duration(chunk_duration)

                    result_text = " ".join(texts) if texts else None

                    # Registrar métricas
                    if self.metrics and start_timestamp:
                        self.metrics.record_chunk_end(
                            start_timestamp, had_transcription=(result_text is not None)
                        )

                    return result_text

                except Exception as e:
                    logger.error(f"Error in MLX transcription: {e}", exc_info=True)
                    if self.metrics:
                        self.metrics.record_error()
                        if start_timestamp:
                            self.metrics.record_chunk_end(start_timestamp, had_transcription=False)
                    return None

        except Exception as e:
            logger.error(f"Error processing WAV chunk: {e}", exc_info=True)
            if self.metrics:
                self.metrics.record_error()
                if start_timestamp:
                    self.metrics.record_chunk_end(start_timestamp, had_transcription=False)
            return None

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

            logger.info(f"Transcription: [{timestamp}] {text}")
            if self.realtime_output:
                print(f"[{timestamp}] {text}", flush=True)

            if self.output_file:
                self._append_to_file(text, absolute_start, absolute_start + (end_time - start_time))

    def _append_to_file(self, text: str, start_time: float, end_time: float) -> None:
        """Escribe transcripción a archivo en tiempo real."""
        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%H:%M:%S", time.localtime(start_time))
            with self.output_file.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {text}\n")
                f.flush()
        except Exception as e:
            logger.error(f"Error writing to file: {e}", exc_info=True)

    def get_full_transcript(self) -> str:
        """Obtiene la transcripción completa hasta el momento."""
        with self.lock:
            return " ".join(self.transcription_buffer)

    def export_transcript(self, formats: list[str], output_dir: Path) -> None:
        """Exporta la transcripción en los formatos especificados."""
        from local_transcriber.transcribe.formats import (
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
