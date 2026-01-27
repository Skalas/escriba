from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("local_transcriber.whisper")


def _build_command(input_path: Path, output_dir: Path) -> list[str]:
    """
    Construye el comando Whisper CLI a partir de una plantilla.
    
    Args:
        input_path: Ruta del archivo de audio a transcribir
        output_dir: Directorio donde guardar la transcripción
    
    Returns:
        Lista de argumentos para subprocess
    """
    template = os.getenv(
        "WHISPER_CMD",
        "whisper --model small --language es --output_format txt "
        "--output_dir {output_dir} {input}",
    )
    rendered = template.replace("{input}", str(input_path)).replace(
        "{output_dir}", str(output_dir)
    )
    return shlex.split(rendered)


def transcribe_file(
    input_path: Path,
    output_dir: Path,
    combined_transcript: Optional[Path] = None,
) -> Optional[Path]:
    """
    Transcribe un archivo de audio usando Whisper CLI.
    
    Args:
        input_path: Ruta del archivo de audio
        output_dir: Directorio donde guardar la transcripción
        combined_transcript: Archivo opcional para transcripción combinada
    
    Returns:
        Ruta del archivo de transcripción creado, o None si falla
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(input_path, output_dir)
    LOGGER.info("Whisper command: %s", " ".join(command))
    subprocess.run(command, check=True)

    candidates = [
        output_dir / f"{input_path.stem}.txt",
        output_dir / f"{input_path.name}.txt",
    ]
    expected_txt = next((path for path in candidates if path.exists()), None)
    if expected_txt is not None:
        LOGGER.info("Transcript created: %s", expected_txt)
        if combined_transcript is not None:
            _append_to_combined(expected_txt, combined_transcript, input_path)
            LOGGER.info("Combined updated: %s", combined_transcript)
        return expected_txt
    LOGGER.warning("Transcript not found for: %s", input_path)
    return None


def _append_to_combined(
    transcript_path: Path, combined_path: Path, source_audio: Path
) -> None:
    """
    Agrega una transcripción a un archivo combinado.
    
    Args:
        transcript_path: Ruta del archivo de transcripción individual
        combined_path: Ruta del archivo combinado
        source_audio: Ruta del archivo de audio original (para el header)
    """
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    content = transcript_path.read_text(encoding="utf-8")
    header = f"\n--- {source_audio.name} ---\n"
    with combined_path.open("a", encoding="utf-8") as handle:
        handle.write(header)
        handle.write(content)
        if not content.endswith("\n"):
            handle.write("\n")

