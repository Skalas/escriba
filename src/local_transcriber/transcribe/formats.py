from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def export_to_json(
    segments: list[dict[str, Any]],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Exporta transcripción a formato JSON estructurado.
    
    Args:
        segments: Lista de segmentos con 'start', 'end', 'text'
        output_path: Ruta donde guardar el archivo JSON
        metadata: Metadata adicional (model, language, etc.)
    """
    # Calcular duración total
    duration_seconds = 0.0
    if segments:
        last_segment = segments[-1]
        duration_seconds = last_segment.get("end", 0.0)

    # Metadata por defecto
    default_metadata = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "duration_seconds": duration_seconds,
    }
    if metadata:
        default_metadata.update(metadata)

    # Estructura JSON
    output_data = {
        "metadata": default_metadata,
        "segments": segments,
    }

    # Escribir archivo
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported JSON transcript to: {output_path}")
    except Exception as e:
        logger.error(f"Error exporting JSON: {e}", exc_info=True)
        raise


def export_to_txt(
    segments: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Exporta transcripción a formato TXT simple.
    
    Args:
        segments: Lista de segmentos con 'start', 'end', 'text'
        output_path: Ruta donde guardar el archivo TXT
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for segment in segments:
                start_time = segment.get("start", 0.0)
                text = segment.get("text", "")
                # Formatear timestamp como HH:MM:SS
                hours = int(start_time // 3600)
                minutes = int((start_time % 3600) // 60)
                seconds = int(start_time % 60)
                timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                f.write(f"[{timestamp}] {text}\n")
        logger.info(f"Exported TXT transcript to: {output_path}")
    except Exception as e:
        logger.error(f"Error exporting TXT: {e}", exc_info=True)
        raise
