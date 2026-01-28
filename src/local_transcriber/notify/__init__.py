"""Notification module for sending summaries via messaging platforms."""

from __future__ import annotations

from local_transcriber.notify.telegram import send_telegram_message

__all__ = ["send_telegram_message"]
