"""Simple speaker change detection based on audio characteristics."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerDetector:
    """
    Detector simple de cambios de speaker basado en características de audio.

    No es diarización completa, pero detecta cambios significativos en
    características de audio entre segmentos.
    """

    def __init__(self, threshold: float = 0.3):
        """
        Inicializa el detector.

        Args:
            threshold: Umbral para detectar cambios (0.0-1.0). Valores más bajos
                detectan más cambios.
        """
        self.threshold = threshold
        self.previous_features: Optional[dict[str, float]] = None
        self.speaker_count = 0
        self.current_speaker = "A"

    def extract_features(self, audio_data: bytes) -> dict[str, float]:
        """
        Extrae características de audio de un chunk.

        Args:
            audio_data: Datos de audio en formato PCM int16 (con o sin header WAV)

        Returns:
            Diccionario con características extraídas
        """
        # Saltar header WAV si existe
        if len(audio_data) < 44:
            return {}

        if audio_data[:4] == b"RIFF":
            # Tiene header WAV, saltar 44 bytes
            pcm_data = audio_data[44:]
        else:
            # No tiene header, es PCM puro
            pcm_data = audio_data

        if len(pcm_data) == 0:
            return {}

        try:
            # Convertir a numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)

            if len(audio_array) == 0:
                return {}

            # Convertir a float32 normalizado
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Características básicas
            features = {
                "rms": float(np.sqrt(np.mean(audio_float**2))),  # RMS energy
                "zero_crossing_rate": float(
                    np.mean(np.abs(np.diff(np.sign(audio_float))))
                ),  # Zero crossing rate
                "spectral_centroid": self._spectral_centroid(audio_float),
                "spectral_rolloff": self._spectral_rolloff(audio_float),
            }

            return features
        except Exception as e:
            logger.debug(f"Error extracting audio features: {e}")
            return {}

    def _spectral_centroid(self, audio: np.ndarray) -> float:
        """Calcula el centroide espectral (frecuencia promedio ponderada)."""
        try:
            # FFT
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            frequency = np.fft.rfftfreq(len(audio))

            if np.sum(magnitude) == 0:
                return 0.0

            # Centroide espectral
            centroid = np.sum(frequency * magnitude) / np.sum(magnitude)
            return float(centroid)
        except Exception:
            return 0.0

    def _spectral_rolloff(self, audio: np.ndarray, rolloff_percent: float = 0.85) -> float:
        """Calcula el spectral rolloff (frecuencia donde se concentra el 85% de la energía)."""
        try:
            # FFT
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            frequency = np.fft.rfftfreq(len(audio))

            # Calcular energía acumulada
            cumsum = np.cumsum(magnitude)
            total_energy = cumsum[-1]

            if total_energy == 0:
                return 0.0

            # Encontrar frecuencia donde se alcanza el rolloff_percent de energía
            target_energy = total_energy * rolloff_percent
            rolloff_idx = np.where(cumsum >= target_energy)[0]

            if len(rolloff_idx) > 0:
                return float(frequency[rolloff_idx[0]])
            return 0.0
        except Exception:
            return 0.0

    def detect_change(self, audio_data: bytes) -> Optional[str]:
        """
        Detecta si hay un cambio de speaker.

        Args:
            audio_data: Datos de audio del chunk actual

        Returns:
            Etiqueta del speaker actual si hay cambio, None si no hay cambio
        """
        current_features = self.extract_features(audio_data)

        if not current_features:
            return None

        if self.previous_features is None:
            # Primer chunk, guardar características
            self.previous_features = current_features
            return f"Speaker {self.current_speaker}"

        # Calcular distancia entre características
        distance = self._feature_distance(self.previous_features, current_features)

        if distance > self.threshold:
            # Cambio detectado
            self.speaker_count += 1
            # Alternar entre A y B (simple)
            if self.current_speaker == "A":
                self.current_speaker = "B"
            else:
                self.current_speaker = "A"

            self.previous_features = current_features
            return f"Speaker {self.current_speaker}"

        # No hay cambio, actualizar características
        self.previous_features = current_features
        return None

    def _feature_distance(self, features1: dict[str, float], features2: dict[str, float]) -> float:
        """
        Calcula la distancia entre dos conjuntos de características.

        Args:
            features1: Primer conjunto de características
            features2: Segundo conjunto de características

        Returns:
            Distancia normalizada (0.0-1.0)
        """
        if not features1 or not features2:
            return 0.0

        # Normalizar características para comparación
        # Usar diferencia relativa para cada feature
        distances = []

        for key in features1:
            if key in features2:
                val1 = features1[key]
                val2 = features2[key]

                if val1 == 0 and val2 == 0:
                    continue

                # Diferencia relativa
                if val1 != 0:
                    rel_diff = abs((val2 - val1) / val1)
                else:
                    rel_diff = abs(val2)

                distances.append(rel_diff)

        if not distances:
            return 0.0

        # Promedio de distancias normalizado
        avg_distance = np.mean(distances)
        # Normalizar a 0-1 (ajustar según necesidad)
        normalized = min(avg_distance, 1.0)

        return float(normalized)

    def reset(self):
        """Reinicia el detector."""
        self.previous_features = None
        self.speaker_count = 0
        self.current_speaker = "A"


# Instancia global (puede ser compartida)
_global_detector: Optional[SpeakerDetector] = None


def detect_speaker_change(audio_data: bytes, threshold: float = 0.3) -> Optional[str]:
    """
    Detecta cambios de speaker en un chunk de audio.

    Args:
        audio_data: Datos de audio en formato PCM int16 o WAV
        threshold: Umbral para detección (0.0-1.0)

    Returns:
        Etiqueta del speaker si hay cambio, None si no hay cambio
    """
    global _global_detector

    if _global_detector is None:
        _global_detector = SpeakerDetector(threshold=threshold)
    elif _global_detector.threshold != threshold:
        # Recrear si el threshold cambió
        _global_detector = SpeakerDetector(threshold=threshold)

    return _global_detector.detect_change(audio_data)
