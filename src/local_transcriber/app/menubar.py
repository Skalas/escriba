"""macOS menu bar app for Local Transcriber."""

from __future__ import annotations

import logging
import webbrowser

import rumps

from local_transcriber.app.database import Database
from local_transcriber.app.server import PORT, start_server
from local_transcriber.app.session import TranscriptionSession
from local_transcriber.config import AppConfig

logger = logging.getLogger(__name__)


def _notify(title: str, subtitle: str, message: str):
    """Send a notification, silently ignoring errors (e.g. missing Info.plist)."""
    try:
        rumps.notification(title, subtitle, message)
    except RuntimeError:
        logger.debug("Notification failed (missing Info.plist), skipping")


class TranscriberMenuBar(rumps.App):
    """Menu bar app that controls transcription sessions."""

    def __init__(self, config: AppConfig):
        super().__init__("LT", quit_button=None)
        self.config = config
        self.db = Database()
        self.app_state: dict = {"config": config, "session": None, "db": self.db}
        self.server = None

        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.toggle_recording),
            None,
            rumps.MenuItem("Open Dashboard", callback=self.open_dashboard),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

    def toggle_recording(self, sender):
        session: TranscriptionSession | None = self.app_state.get("session")

        if session and session.is_active:
            session.stop()
            sender.title = "Start Recording"
            self.title = "LT"
            _notify("Local Transcriber", "Recording stopped", "Transcript saved.")
        else:
            session = TranscriptionSession(self.config, database=self.db)
            session.start()
            self.app_state["session"] = session

            if session.error:
                _notify("Local Transcriber", "Error", session.error)
                return

            sender.title = "Stop Recording"
            self.title = "LT ●"
            _notify("Local Transcriber", "Recording started", "Capturing system audio...")

    def open_dashboard(self, _):
        webbrowser.open(f"http://127.0.0.1:{PORT}")

    def quit_app(self, _):
        session: TranscriptionSession | None = self.app_state.get("session")
        if session and session.is_active:
            session.stop()
        if self.server:
            self.server.shutdown()
        self.db.close()
        rumps.quit_application()


def run_menubar_app(config: AppConfig | None = None):
    """Launch the menu bar app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from dotenv import load_dotenv

    load_dotenv()

    if config is None:
        config = AppConfig.load()

    logger.info(
        "Config: backend=%s, model=%s", config.streaming.backend, config.streaming.model_size
    )
    app = TranscriberMenuBar(config)
    app.server = start_server(app.app_state)
    logger.info("Dashboard at http://127.0.0.1:%s", PORT)
    app.run()
