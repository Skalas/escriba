from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from local_transcriber.audio.call_detection import (
    wait_for_call_start,
)
from local_transcriber.audio.device_detection import (
    list_audio_devices,
    auto_detect_devices,
)
from local_transcriber.audio.live_capture import run_live_capture, run_streaming_capture
from local_transcriber.config import AppConfig, config_to_dict
from local_transcriber.daemon.client import DaemonClient, is_daemon_running
from local_transcriber.daemon.server import run_daemon
from local_transcriber.watch.watch_folder import watch_folder


logger = logging.getLogger(__name__)


app = typer.Typer(
    name="local-transcriber",
    add_completion=False,
    invoke_without_command=True,
    help="Local system + mic audio transcription (macOS).",
)


@app.callback()
def _main_callback(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Path to local-transcriber TOML config.",
        exists=False,
        dir_okay=False,
        file_okay=True,
        readable=True,
    ),
    print_config: bool = typer.Option(
        False,
        "--print-config",
        help="Print resolved config as JSON and exit.",
    ),
) -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = AppConfig.load(config)
    ctx.obj = {"config": cfg}

    if print_config:
        print(json.dumps(config_to_dict(cfg), indent=2, ensure_ascii=False))
        raise typer.Exit(code=0)


def _get_cfg(ctx: typer.Context) -> AppConfig:
    cfg = (ctx.obj or {}).get("config")
    if not isinstance(cfg, AppConfig):
        # Should not happen, but keep a safe fallback.
        cfg = AppConfig.load(None)
    return cfg


@app.command("live")
def cmd_live(
    ctx: typer.Context,
    output_dir: Path = typer.Option(
        Path("transcripts"),
        "--output-dir",
        help="Directorio donde guardar transcripciones.",
    ),
    combined: Path | None = typer.Option(
        None, "--combined", help="Archivo para transcripción combinada."
    ),
) -> None:
    cfg = _get_cfg(ctx)
    run_live_capture(output_dir, combined, config=cfg)


@app.command("live-stream")
def cmd_live_stream(
    ctx: typer.Context,
    auto_start: bool = typer.Option(
        False,
        "--auto-start",
        help="Esperar automáticamente a que inicie una llamada antes de comenzar la transcripción.",
    ),
    output_dir: Path = typer.Option(
        Path("transcripts"),
        "--output-dir",
        help="Directorio donde guardar transcripciones.",
    ),
    combined: Path | None = typer.Option(
        None, "--combined", help="Archivo para transcripción combinada en tiempo real."
    ),
    model_size: str | None = typer.Option(
        None,
        "--model-size",
        help="Tamaño del modelo (tiny, base, small, medium, large).",
    ),
    chunk_duration: float | None = typer.Option(
        None,
        "--chunk-duration",
        help="Duración de chunks en segundos (default: 30.0).",
        min=0.5,
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Idioma para transcripción (código ISO 639-1).",
    ),
    no_realtime_output: bool = typer.Option(
        False,
        "--no-realtime-output",
        help="No mostrar transcripciones en consola en tiempo real.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Dispositivo a usar para transcripción (auto|cpu|cuda).",
    ),
    format: str | None = typer.Option(
        None,
        "--format",
        help="Formatos de exportación separados por comas (txt,json). Ejemplo: txt,json.",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Backend: faster-whisper|openai-whisper|mps|mlx-whisper.",
    ),
    metrics: bool = typer.Option(
        False,
        "--metrics",
        help="Mostrar métricas de captura y transcripción al finalizar.",
    ),
    summarize: bool = typer.Option(
        False,
        "--summarize",
        help="Generar resumen automático con LLM al finalizar.",
    ),
    summary_model: str | None = typer.Option(
        None,
        "--summary-model",
        help="Modelo LLM para resumen (gemini o claude).",
    ),
    speaker_detection: bool = typer.Option(
        False,
        "--speaker-detection",
        help="(Legacy) Habilitar detección simple de cambios de speaker.",
    ),
) -> None:
    cfg = _get_cfg(ctx)

    # Auto-start when call detected
    if auto_start:
        print("Waiting for call to start...")
        app_name = wait_for_call_start()
        if app_name:
            print(f"Call detected in {app_name}. Starting transcription...")
        else:
            print("No call detected. Starting anyway...")

    # Apply CLI overrides (flags > config > env)
    overrides: dict[str, object] = {}
    if model_size:
        overrides["model_size"] = model_size
    if chunk_duration is not None:
        overrides["chunk_duration"] = chunk_duration
    if language:
        overrides["language"] = language
    if no_realtime_output:
        overrides["realtime_output"] = False
    if device:
        overrides["device"] = device
    if format:
        overrides["export_formats"] = [
            v.strip() for v in format.split(",") if v.strip()
        ]
    if backend:
        overrides["backend"] = backend
    if metrics:
        overrides["show_metrics"] = True
    if summarize:
        overrides["summarize"] = True
    if summary_model:
        overrides["summary_model"] = summary_model
    if speaker_detection:
        # Keep compatibility with previous flag: enables simple mode.
        overrides["speaker_mode"] = "simple"

    run_streaming_capture(
        output_dir,
        combined_transcript=combined,
        config=cfg,
        streaming_overrides=overrides,
    )


@app.command("watch")
def cmd_watch(
    input_dir: Path = typer.Option(
        Path("audios"), "--dir", help="Directorio a observar para nuevos audios."
    ),
    output_dir: Path = typer.Option(
        Path("transcripts"),
        "--output-dir",
        help="Directorio donde guardar transcripciones.",
    ),
    combined: Path | None = typer.Option(
        None, "--combined", help="Archivo para transcripción combinada."
    ),
) -> None:
    watch_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        combined_transcript=combined,
    )


@app.command("list-devices")
def cmd_list_devices() -> None:
    print("Detecting audio devices...\n")
    devices = list_audio_devices()

    if devices["inputs"]:
        print("Available audio input devices:")
        for device in devices["inputs"]:
            print(f"  [{device['index']}] {device['name']} (type: {device['type']})")
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


daemon_app = typer.Typer(name="daemon", help="Gestiona el daemon de transcripción.")
app.add_typer(daemon_app, name="daemon")


@daemon_app.command("start")
def daemon_start(
    model_size: str = typer.Option(
        "base", "--model-size", help="Tamaño del modelo a cargar."
    ),
) -> None:
    if is_daemon_running():
        print("Daemon is already running.")
        raise typer.Exit(code=0)
    print(f"Starting daemon with model: {model_size}")
    run_daemon(model_size=model_size)


@daemon_app.command("stop")
def daemon_stop() -> None:
    if not is_daemon_running():
        print("Daemon is not running.")
        raise typer.Exit(code=0)
    client = DaemonClient()
    response = client.stop_daemon()
    if response.get("success"):
        print("Daemon stopped.")
    else:
        raise typer.Exit(code=1)


@daemon_app.command("status")
def daemon_status() -> None:
    if not is_daemon_running():
        print("Daemon is not running.")
        raise typer.Exit(code=0)
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
        raise typer.Exit(code=1)


@daemon_app.command("start-recording")
def daemon_start_recording(
    output_dir: Path = typer.Option(Path("transcripts"), "--output-dir"),
    combined: Path | None = typer.Option(None, "--combined"),
) -> None:
    if not is_daemon_running():
        print("Daemon is not running. Start it with: local-transcriber daemon start")
        raise typer.Exit(code=1)
    client = DaemonClient()
    response = client.start_recording(
        output_dir=str(output_dir), combined=str(combined) if combined else None
    )
    if response.get("success"):
        print(f"Recording started. Output: {output_dir}")
    else:
        raise typer.Exit(code=1)


@daemon_app.command("stop-recording")
def daemon_stop_recording() -> None:
    if not is_daemon_running():
        print("Daemon is not running.")
        raise typer.Exit(code=0)
    client = DaemonClient()
    response = client.stop_recording()
    if response.get("success"):
        print("Recording stopped.")
    else:
        raise typer.Exit(code=1)


@app.command("watch-calendar")
def cmd_watch_calendar(
    auto_start: bool = typer.Option(
        False,
        "--auto-start",
        help="Auto-iniciar transcripción cuando detecte una reunión.",
    ),
    check_interval: int = typer.Option(
        60, "--check-interval", help="Intervalo en segundos para verificar calendario."
    ),
) -> None:
    from local_transcriber.calendar.apple_calendar import watch_calendar

    def on_meeting_detected(event: dict[str, object]) -> None:
        print(f"Meeting detected: {event.get('title', 'Unknown')}")
        if auto_start:
            logger.info(
                "Would start transcription for: %s", event.get("title", "Unknown")
            )

    print("Watching calendar for meetings...")
    print("Press Ctrl+C to stop")
    watch_calendar(on_meeting_detected, check_interval=check_interval)

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping calendar watch...")


@app.command("create-issues")
def cmd_create_issues(
    transcript: Path = typer.Option(
        ..., "--transcript", help="Ruta al archivo de transcripción."
    ),
    repo: str = typer.Option(
        ..., "--repo", help="Repositorio en formato 'owner/repo'."
    ),
    model: str = typer.Option(
        "gemini", "--model", help="Modelo LLM (gemini o claude)."
    ),
) -> None:
    from local_transcriber.integrations.github import create_issues_from_transcript

    if not transcript.exists():
        raise typer.BadParameter(f"Transcript file not found: {transcript}")

    print(f"Creating issues from transcript: {transcript}")
    print(f"Repository: {repo}")

    issues = create_issues_from_transcript(
        transcript_path=transcript,
        repo=repo,
        model=model,
    )

    if issues:
        print(f"\nCreated {len(issues)} issue(s):")
        for issue in issues:
            print(f"  #{issue['number']}: {issue['title']}")
            print(f"    {issue['url']}")
    else:
        print("No issues created.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
