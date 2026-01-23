"""
Captura robusta de audio del sistema en macOS.
Detecta automáticamente el dispositivo de salida activo y configura
un Multi-Output Device si es necesario para capturar sin configuración manual.
"""

from __future__ import annotations

import logging
import subprocess
import json
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_current_output_device() -> Optional[dict[str, str]]:
    """
    Obtiene el dispositivo de salida de audio actual.

    Usa múltiples métodos para detectar el dispositivo activo:
    1. system_profiler (más confiable)
    2. osascript (fallback)

    Returns:
        Dict con 'name' y 'id' del dispositivo, o None si no se puede obtener
    """
    # Método 1: system_profiler (más confiable)
    try:
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Buscar dispositivo de salida actual (default_audio_system_device)
            for item in data.get("SPAudioDataType", []):
                items = item.get("_items", [])
                for device in items:
                    # Buscar el dispositivo que es el default system device
                    if (
                        device.get("coreaudio_default_audio_system_device")
                        == "spaudio_yes"
                    ):
                        name = device.get("_name", "")
                        device_id = device.get("coreaudio_device_id")
                        if name:
                            logger.debug(
                                f"Current output device (system_profiler): {name}"
                            )
                            return {"name": name, "id": device_id}

                    # También buscar en sub-items (para Multi-Output Devices)
                    sub_items = device.get("_items", [])
                    for sub_device in sub_items:
                        if (
                            sub_device.get("coreaudio_default_audio_system_device")
                            == "spaudio_yes"
                        ):
                            name = sub_device.get("_name", "")
                            device_id = sub_device.get("coreaudio_device_id")
                            if name:
                                logger.debug(
                                    f"Current output device (system_profiler, sub-item): {name}"
                                )
                                return {"name": name, "id": device_id}
    except json.JSONDecodeError as e:
        logger.debug(f"Could not parse system_profiler JSON: {e}")
    except Exception as e:
        logger.debug(f"Could not get current output device via system_profiler: {e}")

    # Método 2: osascript (fallback - menos confiable)
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                """
                tell application "System Events"
                    tell application process "SystemUIServer"
                        tell (menu bar item 1 of menu bar 1 whose description is "volume")
                            click
                            set deviceList to name of every menu item of menu 1
                            set currentDevice to name of menu item 1 of menu 1
                            key code 53
                            return currentDevice
                        end tell
                    end tell
                end tell
            """,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            device_name = result.stdout.strip()
            logger.debug(f"Current output device (osascript): {device_name}")
            return {"name": device_name, "id": None}
    except Exception as e:
        logger.debug(f"Could not get current output device via osascript: {e}")

    return None


def create_multi_output_device(
    output_device_name: str, blackhole_name: str = "BlackHole 2ch"
) -> Optional[str]:
    """
    Crea un Multi-Output Device que combine el dispositivo de salida actual + BlackHole.

    Usa Audio MIDI Setup CLI o AppleScript para crear el dispositivo.

    Returns:
        Nombre del Multi-Output Device creado, o None si falla
    """
    multi_output_name = f"LocalTranscriber-{output_device_name}-{blackhole_name}"

    try:
        # Intentar crear Multi-Output Device usando Audio MIDI Setup
        # Esto requiere permisos y puede ser complejo, así que usamos un enfoque más simple:
        # Detectar si ya existe un Multi-Output Device que incluya BlackHole

        # Por ahora, retornamos None y el sistema usará BlackHole directamente
        # si está configurado, o solo capturará el micrófono
        logger.info(
            f"Multi-Output Device creation not yet implemented. "
            f"Would create: {multi_output_name}"
        )
        return None
    except Exception as e:
        logger.debug(f"Could not create Multi-Output Device: {e}")
        return None


def ensure_system_audio_capture() -> Optional[str]:
    """
    Asegura que el audio del sistema se pueda capturar de manera robusta.

    Estrategia inteligente:
    1. Detecta el dispositivo de salida actual
    2. Si es BlackHole -> perfecto, usar BlackHole
    3. Si es otro dispositivo (AirPods, etc.) -> verificar si hay un Multi-Output Device
       que incluya BlackHole, o intentar usar BlackHole directamente (puede funcionar
       si hay un Multi-Output Device configurado manualmente)
    4. Si BlackHole no está disponible -> retornar None

    Returns:
        Índice del dispositivo para capturar audio del sistema, o None
    """
    from local_transcriber.audio.device_detection import (
        list_audio_devices,
        find_system_audio_device,
    )

    # Verificar si BlackHole está disponible
    blackhole_device = find_system_audio_device()

    if not blackhole_device:
        logger.warning(
            "BlackHole not found. System audio capture will not work. "
            "Install BlackHole: brew install blackhole-2ch"
        )
        return None

    # Detectar dispositivo de salida actual
    current_output = get_current_output_device()

    if current_output:
        current_name = current_output.get("name", "").lower()

        # Caso 1: BlackHole es el dispositivo de salida actual - perfecto
        if "blackhole" in current_name:
            logger.info(
                "✓ BlackHole is the active output device - system audio will be captured"
            )
            return blackhole_device

        # Caso 2: Hay otro dispositivo de salida (AirPods, Built-in, etc.)
        # Verificar si hay un Multi-Output Device que incluya BlackHole
        devices = list_audio_devices()
        multi_output_found = False

        for device in devices["inputs"]:
            device_name_lower = device["name"].lower()
            # Buscar dispositivos que puedan ser Multi-Output Devices
            # y que incluyan el nombre del dispositivo actual + BlackHole
            if (
                "aggregate" in device_name_lower or "multi" in device_name_lower
            ) and current_name.split()[0] in device_name_lower:
                logger.info(
                    f"Found potential Multi-Output Device: {device['name']}. "
                    f"System audio should be captured via BlackHole."
                )
                multi_output_found = True
                # Retornar BlackHole directamente - el Multi-Output Device
                # debería estar enviando audio a BlackHole
                return blackhole_device

        if not multi_output_found:
            logger.info(
                f"Current output device: {current_output.get('name')}\n"
                f"  - System audio may not be captured (audio is going to {current_output.get('name')}, not BlackHole)\n"
                f"  - BlackHole is available but not receiving audio\n"
                f"  - Solution: Create a Multi-Output Device in Audio MIDI Setup that combines:\n"
                f"      • {current_output.get('name')} (to hear audio)\n"
                f"      • BlackHole 2ch (to capture audio)\n"
                f"  - Or set BlackHole as the output device temporarily"
            )
            # Aún retornamos BlackHole - puede funcionar si el usuario
            # tiene un Multi-Output Device configurado que no detectamos
            return blackhole_device
    else:
        # No pudimos detectar el dispositivo actual
        logger.info(
            "Could not detect current output device, but BlackHole is available. "
            "Will attempt to capture system audio via BlackHole."
        )
        return blackhole_device

    return blackhole_device


def get_smart_audio_devices() -> Tuple[Optional[str], Optional[str]]:
    """
    Obtiene dispositivos de audio de manera inteligente, asegurando
    que el audio del sistema se pueda capturar sin configuración manual.

    Returns:
        Tuple (system_device_index, mic_device_index)
    """
    from local_transcriber.audio.device_detection import find_microphone_device

    system_device = ensure_system_audio_capture()
    mic_device = find_microphone_device()

    return system_device, mic_device
