"""Telegram notification support."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def send_telegram_message(
    message: str,
    chat_id: str | None = None,
    bot_token: str | None = None,
) -> bool:
    """
    Envía un mensaje a Telegram.

    Args:
        message: Mensaje a enviar
        chat_id: ID del chat (o usar TELEGRAM_CHAT_ID env var)
        bot_token: Token del bot (o usar TELEGRAM_BOT_TOKEN env var)

    Returns:
        True si se envió exitosamente, False en caso contrario
    """
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")

    if not chat_id or not bot_token:
        logger.error("TELEGRAM_CHAT_ID and TELEGRAM_BOT_TOKEN must be set")
        return False

    try:
        import requests

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()

        logger.info("Telegram message sent successfully")
        return True

    except ImportError:
        logger.error("requests not installed. Install with: pip install requests")
        return False
    except Exception as e:
        logger.error("Error sending Telegram message: %s", e, exc_info=True)
        return False


def send_summary(summary_path: Path, chat_id: str | None = None) -> bool:
    """
    Envía un resumen desde un archivo JSON a Telegram.

    Args:
        summary_path: Ruta al archivo JSON del resumen
        chat_id: ID del chat (opcional)

    Returns:
        True si se envió exitosamente
    """
    import json

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        # Formatear mensaje
        message_parts = [f"*📝 Resumen de Reunión*\n"]

        if "summary" in summary:
            message_parts.append(f"*Resumen:*\n{summary['summary']}\n")

        if "key_points" in summary and summary["key_points"]:
            message_parts.append("*Puntos Clave:*")
            for point in summary["key_points"][:5]:  # Limitar a 5
                message_parts.append(f"• {point}")

        if "action_items" in summary and summary["action_items"]:
            message_parts.append("\n*Action Items:*")
            for item in summary["action_items"][:5]:  # Limitar a 5
                task = item.get("task", "")
                assignee = item.get("assignee", "")
                if assignee:
                    message_parts.append(f"• {task} (@{assignee})")
                else:
                    message_parts.append(f"• {task}")

        message = "\n".join(message_parts)
        return send_telegram_message(message, chat_id=chat_id)

    except Exception as e:
        logger.error("Error sending summary to Telegram: %s", e, exc_info=True)
        return False
