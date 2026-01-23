from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from local_transcriber.audio.live_capture import run_live_capture
from local_transcriber.watch.watch_folder import watch_folder


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="local-transcriber")
    subparsers = parser.add_subparsers(dest="command", required=True)

    live_parser = subparsers.add_parser("live", help="Captura en vivo (sistema + mic).")
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

    args = parser.parse_args()
    combined_path = Path(args.combined) if args.combined else None

    if args.command == "live":
        run_live_capture(Path(args.output_dir), combined_path)
    elif args.command == "watch":
        watch_folder(
            input_dir=Path(args.dir),
            output_dir=Path(args.output_dir),
            combined_transcript=combined_path,
        )


if __name__ == "__main__":
    main()

