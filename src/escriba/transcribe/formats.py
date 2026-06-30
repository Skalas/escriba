from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Session-export builders (shared by app layer and knowledge adapters)
# ---------------------------------------------------------------------------


def _segment_speaker_label(segment: dict[str, Any]) -> str | None:
    """Resolved speaker label for display/export (custom name or raw)."""
    display = segment.get("speaker_display")
    if display:
        return str(display)
    raw = segment.get("speaker")
    return str(raw) if raw else None


def _segments_to_transcript(segments: list[dict[str, Any]]) -> str:
    """Build transcript text using display speaker names when available."""
    parts: list[str] = []
    for seg in segments:
        text = seg.get("text") or ""
        speaker = _segment_speaker_label(seg)
        if speaker:
            parts.append(f"[{speaker}] {text}")
        else:
            parts.append(text)
    return " ".join(parts)


def format_export_timestamp(seconds: float | int | None) -> str:
    """Format segment start time as HH:MM:SS for export."""
    total = int(seconds or 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_session_export_markdown(
    session: dict[str, Any], segments: list[dict[str, Any]]
) -> str:
    """Build a Markdown export bundle for a session."""
    lines: list[str] = [f"# {session.get('name', 'Session')}", ""]

    metadata: list[str] = []
    if session.get("started_at"):
        metadata.append(f"**Date:** {session['started_at']}")
    duration = session.get("duration_seconds")
    if duration is not None:
        metadata.append(f"**Duration:** {format_export_timestamp(duration)}")
    if metadata:
        lines.extend(metadata)
        lines.append("")

    notes_text = (session.get("notes_text") or "").strip()
    if notes_text:
        lines.extend(["## Notes", "", notes_text, ""])

    lines.extend(["## Transcript", ""])
    for seg in segments:
        seg_id = seg.get("id")
        anchor = ""
        if seg_id is not None:
            anchor = f'<a id="seg-{int(seg_id)}"></a>'
        timestamp = format_export_timestamp(seg.get("start_time"))
        text = seg.get("text") or ""
        speaker = _segment_speaker_label(seg)
        if speaker:
            safe_speaker = speaker.replace("*", "\\*")
            lines.append(f"{anchor}[{timestamp}] **{safe_speaker}**: {text}")
        else:
            lines.append(f"{anchor}[{timestamp}] {text}")

    return "\n".join(lines)


def build_session_export_txt(
    session: dict[str, Any], segments: list[dict[str, Any]]
) -> str:
    """Build a plain-text export bundle for a session."""
    lines: list[str] = [session.get("name", "Session"), ""]

    if session.get("started_at"):
        lines.append(f"Date: {session['started_at']}")
    duration = session.get("duration_seconds")
    if duration is not None:
        lines.append(f"Duration: {format_export_timestamp(duration)}")
    if session.get("started_at") or duration is not None:
        lines.append("")

    notes_text = (session.get("notes_text") or "").strip()
    if notes_text:
        lines.extend(["Notes", notes_text, ""])

    lines.append("Transcript")
    for seg in segments:
        timestamp = format_export_timestamp(seg.get("start_time"))
        text = seg.get("text") or ""
        speaker = _segment_speaker_label(seg)
        if speaker:
            lines.append(f"{timestamp}  [{speaker}] {text}")
        else:
            lines.append(f"{timestamp}  {text}")

    return "\n".join(lines)

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
        logger.info("Exported JSON transcript to: %s", output_path)
    except Exception as e:
        logger.error("Error exporting JSON: %s", e, exc_info=True)
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
                # Formatear timestamp como HH:MM:SS.mmm (con milisegundos para precisión)
                hours = int(start_time // 3600)
                minutes = int((start_time % 3600) // 60)
                seconds = int(start_time % 60)
                millis = int((start_time % 1) * 1000)
                timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
                f.write(f"[{timestamp}] {text}\n")
        logger.info("Exported TXT transcript to: %s", output_path)
    except Exception as e:
        logger.error("Error exporting TXT: %s", e, exc_info=True)
        raise


def _format_timestamp_srt(seconds: float) -> str:
    """
    Formatea un timestamp en formato SRT (HH:MM:SS,mmm).

    Args:
        seconds: Tiempo en segundos

    Returns:
        String formateado como HH:MM:SS,mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_to_srt(
    segments: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Exporta transcripción a formato SRT (subtítulos).

    Args:
        segments: Lista de segmentos con 'start', 'end', 'text'
        output_path: Ruta donde guardar el archivo SRT
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for index, segment in enumerate(segments, start=1):
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", start_time)
                text = segment.get("text", "").strip()

                if not text:
                    continue

                # Formatear timestamps SRT
                start_str = _format_timestamp_srt(start_time)
                end_str = _format_timestamp_srt(end_time)

                # Escribir entrada SRT
                f.write(f"{index}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{text}\n")
                f.write("\n")

        logger.info("Exported SRT transcript to: %s", output_path)
    except Exception as e:
        logger.error("Error exporting SRT: %s", e, exc_info=True)
        raise


def _format_timestamp_markdown(seconds: float) -> str:
    """
    Formatea un timestamp en formato Markdown (HH:MM:SS).

    Args:
        seconds: Tiempo en segundos

    Returns:
        String formateado como HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def export_to_markdown(
    segments: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Exporta transcripción a formato Markdown.

    Args:
        segments: Lista de segmentos con 'start', 'end', 'text'
        output_path: Ruta donde guardar el archivo Markdown
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            # Header principal
            f.write("# Transcripción\n\n")

            # Escribir cada segmento como sección
            for segment in segments:
                start_time = segment.get("start", 0.0)
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker")

                if not text:
                    continue

                # Formatear timestamp
                timestamp = _format_timestamp_markdown(start_time)

                # Escribir como sección con timestamp
                if speaker:
                    f.write(f"## [{timestamp}] [{speaker}] {text}\n\n")
                else:
                    f.write(f"## [{timestamp}] {text}\n\n")

        logger.info("Exported Markdown transcript to: %s", output_path)
    except Exception as e:
        logger.error("Error exporting Markdown: %s", e, exc_info=True)
        raise
