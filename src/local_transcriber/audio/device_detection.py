"""
Detección automática de dispositivos de audio en macOS.
Similar a Notion AI, detecta automáticamente los dispositivos sin configuración manual.
"""

from __future__ import annotations

import logging
import subprocess
import re
from typing import Optional

logger = logging.getLogger(__name__)


def list_audio_devices() -> dict[str, list[dict[str, str]]]:
    """
    Lista todos los dispositivos de audio disponibles usando ffmpeg.

    Returns:
        Dict con 'inputs' y 'outputs', cada uno con lista de dispositivos
        con 'index', 'name', y 'type'
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # Parsear salida de ffmpeg
        devices = {"inputs": [], "outputs": []}
        current_section = None

        # Debug: mostrar stderr completo para entender el formato
        logger.debug("Full ffmpeg stderr output:")
        for i, line in enumerate(result.stderr.split("\n")[:50]):
            if line.strip():
                logger.debug(f"  Line {i}: {line}")

        for line in result.stderr.split("\n"):
            line_lower = line.lower()

            # Detectar sección de audio (más flexible)
            if "audio devices" in line_lower and "avfoundation" in line_lower:
                current_section = "inputs"
                logger.debug(f"Found audio devices section: {line}")
                continue
            elif "video devices" in line_lower and "avfoundation" in line_lower:
                current_section = None  # No procesamos video
                continue

            # Buscar líneas con dispositivos - múltiples formatos posibles
            if current_section == "inputs":
                # Formato 1: [0] Device Name
                match = re.search(r"\[(\d+)\]\s+(.+)", line)
                if match:
                    index = match.group(1)
                    name = match.group(2).strip()
                else:
                    # Formato 2: Buscar nombres de dispositivos conocidos en la línea
                    # Si la línea contiene palabras clave de dispositivos
                    device_keywords = [
                        "airpods",
                        "built-in",
                        "microphone",
                        "iphone",
                        "ipad",
                        "output",
                    ]
                    if any(keyword in line_lower for keyword in device_keywords):
                        # Intentar extraer el nombre
                        # Buscar después de "]" o al final de la línea
                        name_match = re.search(r"(?:\[.*?\]\s*)?(.+)", line)
                        if name_match:
                            name = name_match.group(1).strip()
                            # Si no hay índice, usar contador
                            index_match = re.search(r"\[(\d+)\]", line)
                            index = (
                                index_match.group(1)
                                if index_match
                                else str(len(devices["inputs"]))
                            )
                        else:
                            continue
                    else:
                        continue

                # Limpiar nombre
                name = re.sub(r"^\[in#\d+.*?\]\s*", "", name)
                name = re.sub(r"^\[.*?\]\s*", "", name)
                name = re.sub(r"\s+", " ", name).strip()

                # Si el nombre está vacío o es muy corto, saltar
                if not name or len(name) < 2:
                    continue

                device_type = _classify_device(name)
                # Solo agregar si no está excluido
                if device_type != "excluded":
                    devices["inputs"].append(
                        {
                            "index": index,
                            "name": name,
                            "type": device_type,
                        }
                    )
                    logger.info(f"Found device: [{index}] {name} (type: {device_type})")
                else:
                    logger.debug(f"Excluding device: {name}")

        logger.info(f"Detected {len(devices['inputs'])} audio input devices")
        if devices["inputs"]:
            logger.info("Available devices:")
            for device in devices["inputs"]:
                logger.info(
                    f"  [{device['index']}] {device['name']} ({device['type']})"
                )
        return devices
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}", exc_info=True)
        # Retornar estructura vacía si falla
        return {"inputs": [], "outputs": []}


def _classify_device(name: str) -> str:
    """
    Clasifica un dispositivo de audio por su nombre.
    
    Args:
        name: Nombre del dispositivo
    
    Returns:
        Tipo del dispositivo: "airpods", "microphone", "external_microphone",
        "system_output", "unknown", o "excluded"
    """
    name_lower = name.lower()

    # Excluir dispositivos que no queremos (más estricto)
    if "iphone" in name_lower or "iphone de" in name_lower:
        return "excluded"  # Excluir iPhone (cualquier variante)
    if "ipad" in name_lower:
        return "excluded"  # Excluir iPad

    # Clasificar dispositivos deseados (orden de prioridad)
    if (
        "airpods" in name_lower
        or "airpods pro" in name_lower
        or "airpods max" in name_lower
    ):
        return "airpods"  # Máxima prioridad para AirPods
    elif "built-in" in name_lower and "microphone" in name_lower:
        return "microphone"
    elif "microphone" in name_lower and "built-in" not in name_lower:
        # Micrófonos externos (pero no iPhone/iPad)
        # Verificar que no sea iPhone/iPad antes de clasificar como externo
        if "iphone" not in name_lower and "ipad" not in name_lower:
            return "external_microphone"
        else:
            return "excluded"
    elif "built-in" in name_lower and "output" in name_lower:
        return "system_output"
    else:
        # Si contiene "iphone" o "ipad" en cualquier parte, excluir
        if "iphone" in name_lower or "ipad" in name_lower:
            return "excluded"
        return "unknown"


def find_microphone_device() -> Optional[str]:
    """
    Encuentra automáticamente el micrófono activo.

    Prioridad:
    1. AirPods (si están conectados) - máxima prioridad
    2. Micrófonos externos
    3. Built-in Microphone

    Returns:
        Índice del dispositivo o None si no se encuentra
    """
    devices = list_audio_devices()

    logger.info(f"Searching for microphone among {len(devices['inputs'])} devices...")
    for device in devices["inputs"]:
        logger.debug(
            f"  Checking device: [{device['index']}] {device['name']} (type: {device['type']})"
        )

    # Prioridad 1: Buscar AirPods primero (si están conectados)
    for device in devices["inputs"]:
        if device["type"] == "airpods":
            logger.info(
                f"✓ Selected AirPods microphone: {device['name']} (index {device['index']})"
            )
            return device["index"]

    # Prioridad 2: Buscar micrófonos externos (pero no iPhone/iPad)
    for device in devices["inputs"]:
        if device["type"] == "external_microphone":
            logger.info(
                f"✓ Selected external microphone: {device['name']} (index {device['index']})"
            )
            return device["index"]

    # Prioridad 3: Buscar Built-in Microphone
    for device in devices["inputs"]:
        if device["type"] == "microphone":
            logger.info(
                f"✓ Selected built-in microphone: {device['name']} (index {device['index']})"
            )
            return device["index"]

    # Si no encontramos nada, mostrar todos los dispositivos disponibles para debug
    logger.warning("No microphone device found. Available devices:")
    for device in devices["inputs"]:
        logger.warning(
            f"  [{device['index']}] {device['name']} (type: {device['type']})"
        )
    return None


def auto_detect_devices() -> tuple[Optional[str], Optional[str]]:
    """
    Detecta automáticamente los dispositivos de audio.

    Nota: El audio del sistema ahora se captura usando ScreenCaptureKit,
    así que system_device siempre será None. Solo detectamos el micrófono.

    Returns:
        Tuple (system_device_index, mic_device_index)
        system_device_index será None porque usamos ScreenCaptureKit
    """
    # ScreenCaptureKit captura el audio del sistema directamente
    # No necesitamos un dispositivo virtual
    system_device = None
    mic_device = find_microphone_device()
    return system_device, mic_device
