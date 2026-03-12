from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaptureMetrics:
    """
    Métricas de captura y transcripción.

    Monitorea la calidad de la captura de audio y transcripción en tiempo real.
    """

    chunks_processed: int = 0
    chunks_silent: int = 0
    chunks_with_transcription: int = 0
    errors: int = 0
    total_audio_duration: float = 0.0
    total_transcription_time: float = 0.0
    audio_level_sum: float = 0.0
    audio_level_samples: int = 0

    # Timestamps para calcular latencia
    _chunk_start_times: list[float] = field(default_factory=list)
    _chunk_end_times: list[float] = field(default_factory=list)
    _latencies: list[float] = field(default_factory=list)

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_chunk_start(self) -> float:
        """
        Registra el inicio del procesamiento de un chunk.

        Returns:
            Timestamp del inicio
        """
        timestamp = time.time()
        with self._lock:
            self._chunk_start_times.append(timestamp)
        return timestamp

    def record_chunk_end(
        self, start_timestamp: float, had_transcription: bool = False
    ) -> None:
        """
        Registra el fin del procesamiento de un chunk.

        Args:
            start_timestamp: Timestamp del inicio (retornado por record_chunk_start)
            had_transcription: Si el chunk produjo transcripción
        """
        end_timestamp = time.time()
        latency = (end_timestamp - start_timestamp) * 1000  # Convertir a ms

        with self._lock:
            self.chunks_processed += 1
            if had_transcription:
                self.chunks_with_transcription += 1
            else:
                self.chunks_silent += 1
            self._chunk_end_times.append(end_timestamp)
            self._latencies.append(latency)
            # Mantener solo las últimas 1000 latencias para no usar demasiada memoria
            if len(self._latencies) > 1000:
                self._latencies.pop(0)
                self._chunk_start_times.pop(0)
                self._chunk_end_times.pop(0)

    def record_error(self) -> None:
        """Registra un error en el procesamiento."""
        with self._lock:
            self.errors += 1

    def record_audio_level(self, audio_data: bytes) -> None:
        """
        Registra el nivel de audio de un chunk.

        Args:
            audio_data: Datos de audio en formato PCM int16
        """
        try:
            # Convertir bytes a numpy array (skip WAV header si existe)
            if len(audio_data) < 44:
                return

            # Verificar si tiene header WAV
            if audio_data[:4] == b"RIFF":
                # Tiene header WAV, saltar 44 bytes
                audio_array = np.frombuffer(audio_data[44:], dtype=np.int16)
            else:
                # No tiene header, es PCM puro
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

            if len(audio_array) == 0:
                return

            # Calcular RMS (Root Mean Square) para nivel de audio
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            # Convertir a dB (referencia: máximo int16 = 32767)
            if rms > 0:
                db = 20 * np.log10(rms / 32767.0)
            else:
                db = -np.inf

            with self._lock:
                self.audio_level_sum += db
                self.audio_level_samples += 1
        except Exception as e:
            logger.debug("Error calculating audio level: %s", e)

    def record_transcription_time(self, duration: float) -> None:
        """
        Registra el tiempo que tomó una transcripción.

        Args:
            duration: Duración en segundos
        """
        with self._lock:
            self.total_transcription_time += duration

    def record_audio_duration(self, duration: float) -> None:
        """
        Registra la duración de audio procesado.

        Args:
            duration: Duración en segundos
        """
        with self._lock:
            self.total_audio_duration += duration

    @property
    def avg_latency_ms(self) -> float:
        """Latencia promedio en milisegundos."""
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    @property
    def min_latency_ms(self) -> float:
        """Latencia mínima en milisegundos."""
        with self._lock:
            if not self._latencies:
                return 0.0
            return min(self._latencies)

    @property
    def max_latency_ms(self) -> float:
        """Latencia máxima en milisegundos."""
        with self._lock:
            if not self._latencies:
                return 0.0
            return max(self._latencies)

    @property
    def audio_level_db(self) -> float:
        """Nivel de audio promedio en dB."""
        with self._lock:
            if self.audio_level_samples == 0:
                return -np.inf
            return self.audio_level_sum / self.audio_level_samples

    @property
    def silent_chunks_percent(self) -> float:
        """Porcentaje de chunks silenciosos."""
        with self._lock:
            if self.chunks_processed == 0:
                return 0.0
            return (self.chunks_silent / self.chunks_processed) * 100.0

    @property
    def transcription_chunks_percent(self) -> float:
        """Porcentaje de chunks con transcripción."""
        with self._lock:
            if self.chunks_processed == 0:
                return 0.0
            return (self.chunks_with_transcription / self.chunks_processed) * 100.0

    @property
    def error_rate(self) -> float:
        """Tasa de errores (errores por chunk procesado)."""
        with self._lock:
            if self.chunks_processed == 0:
                return 0.0
            return (self.errors / self.chunks_processed) * 100.0

    def get_summary(self) -> dict[str, any]:
        """
        Obtiene un resumen de las métricas.

        Returns:
            Diccionario con todas las métricas
        """
        with self._lock:
            return {
                "chunks_processed": self.chunks_processed,
                "chunks_silent": self.chunks_silent,
                "chunks_with_transcription": self.chunks_with_transcription,
                "silent_chunks_percent": self.silent_chunks_percent,
                "transcription_chunks_percent": self.transcription_chunks_percent,
                "errors": self.errors,
                "error_rate_percent": self.error_rate,
                "avg_latency_ms": self.avg_latency_ms,
                "min_latency_ms": self.min_latency_ms,
                "max_latency_ms": self.max_latency_ms,
                "audio_level_db": self.audio_level_db,
                "total_audio_duration_seconds": self.total_audio_duration,
                "total_transcription_time_seconds": self.total_transcription_time,
            }

    def try_get_summary(self, timeout_seconds: float = 0.5) -> dict[str, any] | None:
        """
        Obtiene un resumen de métricas sin bloquear indefinidamente.

        Esto evita hangs en shutdown si otro thread mantiene el lock.

        Args:
            timeout_seconds: Tiempo máximo para adquirir el lock.

        Returns:
            Summary dict o None si no se pudo adquirir el lock a tiempo.
        """
        acquired = self._lock.acquire(timeout=timeout_seconds)
        if not acquired:
            return None
        try:
            latencies = list(self._latencies)
            chunks_processed = self.chunks_processed
            chunks_silent = self.chunks_silent
            chunks_with_transcription = self.chunks_with_transcription
            errors = self.errors

            avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else 0.0
            min_latency_ms = min(latencies) if latencies else 0.0
            max_latency_ms = max(latencies) if latencies else 0.0

            silent_chunks_percent = (
                (chunks_silent / chunks_processed) * 100.0 if chunks_processed else 0.0
            )
            transcription_chunks_percent = (
                (chunks_with_transcription / chunks_processed) * 100.0
                if chunks_processed
                else 0.0
            )
            error_rate_percent = (
                (errors / chunks_processed) * 100.0 if chunks_processed else 0.0
            )

            audio_level_db = (
                (self.audio_level_sum / self.audio_level_samples)
                if self.audio_level_samples
                else -np.inf
            )

            return {
                "chunks_processed": chunks_processed,
                "chunks_silent": chunks_silent,
                "chunks_with_transcription": chunks_with_transcription,
                "silent_chunks_percent": silent_chunks_percent,
                "transcription_chunks_percent": transcription_chunks_percent,
                "errors": errors,
                "error_rate_percent": error_rate_percent,
                "avg_latency_ms": avg_latency_ms,
                "min_latency_ms": min_latency_ms,
                "max_latency_ms": max_latency_ms,
                "audio_level_db": audio_level_db,
                "total_audio_duration_seconds": self.total_audio_duration,
                "total_transcription_time_seconds": self.total_transcription_time,
            }
        finally:
            self._lock.release()

    def print_summary(self, timeout_seconds: float = 0.5) -> None:
        """Imprime un resumen de las métricas en consola."""
        summary = self.try_get_summary(timeout_seconds=timeout_seconds)
        if summary is None:
            print(
                "\n[CaptureMetrics] Skipping metrics print: lock busy (shutdown in progress)\n"
            )
            return
        print("\n" + "=" * 60)
        print("📊 Capture Metrics Summary")
        print("=" * 60)
        print(f"Chunks processed:        {summary['chunks_processed']}")
        print(
            f"Chunks with transcription: {summary['chunks_with_transcription']} ({summary['transcription_chunks_percent']:.1f}%)"
        )
        print(
            f"Silent chunks:           {summary['chunks_silent']} ({summary['silent_chunks_percent']:.1f}%)"
        )
        print(
            f"Errors:                  {summary['errors']} ({summary['error_rate_percent']:.2f}%)"
        )
        print("\nLatency:")
        print(f"  Average:               {summary['avg_latency_ms']:.1f} ms")
        print(f"  Min:                    {summary['min_latency_ms']:.1f} ms")
        print(f"  Max:                    {summary['max_latency_ms']:.1f} ms")
        print("\nAudio:")
        print(f"  Average level:         {summary['audio_level_db']:.1f} dB")
        print(
            f"  Total duration:        {summary['total_audio_duration_seconds']:.1f} s"
        )
        print(
            f"  Transcription time:    {summary['total_transcription_time_seconds']:.1f} s"
        )
        print("=" * 60 + "\n")

    def reset(self) -> None:
        """Reinicia todas las métricas."""
        with self._lock:
            self.chunks_processed = 0
            self.chunks_silent = 0
            self.chunks_with_transcription = 0
            self.errors = 0
            self.total_audio_duration = 0.0
            self.total_transcription_time = 0.0
            self.audio_level_sum = 0.0
            self.audio_level_samples = 0
            self._chunk_start_times.clear()
            self._chunk_end_times.clear()
            self._latencies.clear()
