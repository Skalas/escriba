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
from local_transcriber.daemon.client import DaemonClient, is_daemon_running
from local_transcriber.daemon.server import run_daemon
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
        help="Duración de chunks en segundos (default: 30.0). Sobrescribe STREAMING_CHUNK_DURATION.",
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
    live_stream_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Dispositivo a usar para transcripción (auto usa CPU). faster-whisper solo soporta cpu y cuda. Sobrescribe STREAMING_DEVICE.",
    )
    live_stream_parser.add_argument(
        "--format",
        default=None,
        help="Formatos de exportación separados por comas (txt,json). Ejemplo: txt,json. Sobrescribe STREAMING_EXPORT_FORMATS.",
    )
    live_stream_parser.add_argument(
        "--backend",
        choices=["faster-whisper", "openai-whisper", "mps", "mlx-whisper"],
        default=None,
        help="Backend a usar: faster-whisper (CPU optimizado, default), openai-whisper (soporta MPS/GPU), o mlx-whisper (Apple Silicon GPU). Sobrescribe STREAMING_BACKEND.",
    )
    live_stream_parser.add_argument(
        "--metrics",
        action="store_true",
        help="Mostrar métricas de captura y transcripción al finalizar. Sobrescribe STREAMING_SHOW_METRICS.",
    )
    live_stream_parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generar resumen automático con LLM al finalizar. Sobrescribe STREAMING_SUMMARIZE.",
    )
    live_stream_parser.add_argument(
        "--summary-model",
        choices=["gemini", "claude"],
        default=None,
        help="Modelo LLM para resumen (gemini o claude). Sobrescribe STREAMING_SUMMARY_MODEL.",
    )
    live_stream_parser.add_argument(
        "--speaker-detection",
        action="store_true",
        help="Habilitar detección de cambios de speaker. Sobrescribe STREAMING_SPEAKER_DETECTION.",
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

    # Daemon commands
    daemon_parser = subparsers.add_parser(
        "daemon",
        help="Gestiona el daemon de transcripción (modelo pre-cargado).",
    )
    daemon_subparsers = daemon_parser.add_subparsers(dest="daemon_command", required=True)

    daemon_start_parser = daemon_subparsers.add_parser(
        "start", help="Inicia el daemon con modelo pre-cargado."
    )
    daemon_start_parser.add_argument(
        "--model-size",
        default="base",
        help="Tamaño del modelo a cargar (default: base).",
    )

    daemon_subparsers.add_parser("stop", help="Detiene el daemon.")
    daemon_subparsers.add_parser("status", help="Muestra el estado del daemon.")

    start_recording_parser = daemon_subparsers.add_parser(
        "start-recording", help="Inicia una grabación usando el daemon."
    )
    start_recording_parser.add_argument(
        "--output-dir",
        default="transcripts",
        help="Directorio donde guardar transcripciones.",
    )
    start_recording_parser.add_argument(
        "--combined",
        default=None,
        help="Archivo para transcripción combinada.",
    )

    daemon_subparsers.add_parser("stop-recording", help="Detiene la grabación actual.")

    # Calendar watch command
    calendar_parser = subparsers.add_parser(
        "watch-calendar",
        help="Observa el calendario y auto-inicia transcripciones para reuniones.",
    )
    calendar_parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Auto-iniciar transcripción cuando detecte una reunión.",
    )
    calendar_parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Intervalo en segundos para verificar calendario (default: 60).",
    )

    # GitHub issues command
    issues_parser = subparsers.add_parser(
        "create-issues",
        help="Crea GitHub issues desde una transcripción.",
    )
    issues_parser.add_argument(
        "--transcript",
        required=True,
        help="Ruta al archivo de transcripción.",
    )
    issues_parser.add_argument(
        "--repo",
        required=True,
        help="Repositorio en formato 'owner/repo'.",
    )
    issues_parser.add_argument(
        "--model",
        choices=["gemini", "claude"],
        default="gemini",
        help="Modelo LLM para extraer action items (default: gemini).",
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
        if args.device:
            os.environ["STREAMING_DEVICE"] = args.device
        if args.format:
            os.environ["STREAMING_EXPORT_FORMATS"] = args.format
        if args.backend:
            os.environ["STREAMING_BACKEND"] = args.backend
        if args.metrics:
            os.environ["STREAMING_SHOW_METRICS"] = "true"
        if args.summarize:
            os.environ["STREAMING_SUMMARIZE"] = "true"
        if args.summary_model:
            os.environ["STREAMING_SUMMARY_MODEL"] = args.summary_model
        if args.speaker_detection:
            os.environ["STREAMING_SPEAKER_DETECTION"] = "true"

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
            print("  System audio: Using ScreenCaptureKit (no device needed)")
        if mic_device:
            print(f"  Microphone: Device {mic_device}")
        else:
            print("  Microphone: Not found")
    elif args.command == "daemon":
        if args.daemon_command == "start":
            if is_daemon_running():
                print("Daemon is already running.")
                return
            print(f"Starting daemon with model: {args.model_size}")
            run_daemon(model_size=args.model_size)
        elif args.daemon_command == "stop":
            if not is_daemon_running():
                print("Daemon is not running.")
                return
            client = DaemonClient()
            response = client.stop_daemon()
            if response.get("success"):
                print("Daemon stopped.")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
        elif args.daemon_command == "status":
            if not is_daemon_running():
                print("Daemon is not running.")
                return
            client = DaemonClient()
            response = client.status()
            if response.get("success"):
                status = response.get("status", {})
                print("Daemon Status:")
                print(f"  Running: {status.get('running', False)}")
                print(f"  Model loaded: {status.get('model_loaded', False)}")
                print(f"  Model size: {status.get('model_size', 'N/A')}")
                print(f"  Recording: {status.get('recording', False)}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
        elif args.daemon_command == "start-recording":
            if not is_daemon_running():
                print("Daemon is not running. Start it with: local-transcriber daemon start")
                return
            client = DaemonClient()
            response = client.start_recording(
                output_dir=args.output_dir, combined=args.combined
            )
            if response.get("success"):
                print(f"Recording started. Output: {args.output_dir}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
        elif args.daemon_command == "stop-recording":
            if not is_daemon_running():
                print("Daemon is not running.")
                return
            client = DaemonClient()
            response = client.stop_recording()
            if response.get("success"):
                print("Recording stopped.")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
    elif args.command == "watch-calendar":
        from local_transcriber.calendar.apple_calendar import watch_calendar, get_upcoming_events, has_meeting_link
        
        def on_meeting_detected(event):
            print(f"Meeting detected: {event.get('title', 'Unknown')}")
            if args.auto_start:
                print("Auto-starting transcription...")
                # Aquí se iniciaría la transcripción
                # Por ahora solo mostramos el mensaje
                logger.info(f"Would start transcription for: {event.get('title')}")
        
        print("Watching calendar for meetings...")
        print("Press Ctrl+C to stop")
        watch_calendar(on_meeting_detected, check_interval=args.check_interval)
        
        # Mantener el proceso corriendo
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping calendar watch...")
    elif args.command == "create-issues":
        from local_transcriber.integrations.github import create_issues_from_transcript

        transcript_path = Path(args.transcript)
        if not transcript_path.exists():
            print(f"Error: Transcript file not found: {transcript_path}")
            return

        print(f"Creating issues from transcript: {transcript_path}")
        print(f"Repository: {args.repo}")

        issues = create_issues_from_transcript(
            transcript_path=transcript_path,
            repo=args.repo,
            model=args.model,
        )

        if issues:
            print(f"\nCreated {len(issues)} issue(s):")
            for issue in issues:
                print(f"  #{issue['number']}: {issue['title']}")
                print(f"    {issue['url']}")
        else:
            print("No issues created.")


if __name__ == "__main__":
    main()
