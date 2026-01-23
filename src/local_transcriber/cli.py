from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from local_transcriber.audio.call_detection import (
    detect_active_call,
    wait_for_call_start,
)
from local_transcriber.audio.device_detection import (
    list_audio_devices,
    auto_detect_devices,
)
from local_transcriber.audio.live_capture import run_live_capture, run_streaming_capture
from local_transcriber.watch.watch_folder import watch_folder


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="local-transcriber")
    subparsers = parser.add_subparsers(dest="command", required=True)

    live_parser = subparsers.add_parser(
        "live", help="Captura en vivo (sistema + mic) - modo legacy."
    )
    live_parser.add_argument(
        "--output-dir",
        default="transcripts",
        help="Directorio donde guardar transcripciones.",
    )
    live_parser.add_argument(
        "--combined",
        default=None,
        help="Archivo para transcripción combinada.",
    )

    live_stream_parser = subparsers.add_parser(
        "live-stream",
        help="Captura y transcripción en tiempo real (streaming) - similar a Notion AI.",
    )
    live_stream_parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Esperar automáticamente a que inicie una llamada antes de comenzar la transcripción.",
    )
    live_stream_parser.add_argument(
        "--output-dir",
        default="transcripts",
        help="Directorio donde guardar transcripciones.",
    )
    live_stream_parser.add_argument(
        "--combined",
        default=None,
        help="Archivo para transcripción combinada en tiempo real.",
    )
    live_stream_parser.add_argument(
        "--model-size",
        default=None,
        help="Tamaño del modelo (tiny, base, small, medium, large). Sobrescribe STREAMING_MODEL_SIZE.",
    )
    live_stream_parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Duración de chunks en segundos (default: 2.0). Sobrescribe STREAMING_CHUNK_DURATION.",
    )
    live_stream_parser.add_argument(
        "--language",
        default=None,
        help="Idioma para transcripción (código ISO 639-1). Sobrescribe STREAMING_LANGUAGE.",
    )
    live_stream_parser.add_argument(
        "--no-realtime-output",
        action="store_true",
        help="No mostrar transcripciones en consola en tiempo real.",
    )

    watch_parser = subparsers.add_parser("watch", help="Transcribe audios nuevos.")
    watch_parser.add_argument(
        "--dir",
        default="audios",
        help="Directorio a observar para nuevos audios.",
    )
    watch_parser.add_argument(
        "--output-dir",
        default="transcripts",
        help="Directorio donde guardar transcripciones.",
    )
    watch_parser.add_argument(
        "--combined",
        default=None,
        help="Archivo para transcripción combinada.",
    )

    list_devices_parser = subparsers.add_parser(
        "list-devices",
        help="Lista dispositivos de audio disponibles y muestra auto-detección.",
    )

    args = parser.parse_args()

    # combined_path solo existe para algunos comandos
    combined_path = None
    if hasattr(args, "combined") and args.combined:
        combined_path = Path(args.combined)

    if args.command == "live":
        run_live_capture(Path(args.output_dir), combined_path)
    elif args.command == "live-stream":
        # Configurar variables de entorno si se pasaron como argumentos
        if args.model_size:
            os.environ["STREAMING_MODEL_SIZE"] = args.model_size
        if args.chunk_duration is not None:
            os.environ["STREAMING_CHUNK_DURATION"] = str(args.chunk_duration)
        if args.language:
            os.environ["STREAMING_LANGUAGE"] = args.language
        if args.no_realtime_output:
            os.environ["STREAMING_REALTIME_OUTPUT"] = "false"

        # Auto-detección de llamada si está habilitada
        if args.auto_start:
            print("Waiting for call to start...")
            app_name = wait_for_call_start()
            if app_name:
                print(f"Call detected in {app_name}. Starting transcription...")
            else:
                print("No call detected. Starting anyway...")

        run_streaming_capture(Path(args.output_dir), combined_path)
    elif args.command == "watch":
        watch_folder(
            input_dir=Path(args.dir),
            output_dir=Path(args.output_dir),
            combined_transcript=combined_path,
        )
    elif args.command == "list-devices":
        print("Detecting audio devices...\n")
        devices = list_audio_devices()

        if devices["inputs"]:
            print("Available audio input devices:")
            for device in devices["inputs"]:
                print(
                    f"  [{device['index']}] {device['name']} (type: {device['type']})"
                )
        else:
            print("No audio input devices found.")

        print("\nAuto-detection results:")
        system_device, mic_device = auto_detect_devices()
        if system_device:
            print(f"  System audio: Device {system_device}")
        else:
            print("  System audio: Not found (may need BlackHole)")
        if mic_device:
            print(f"  Microphone: Device {mic_device}")
        else:
            print("  Microphone: Not found")


if __name__ == "__main__":
    main()
