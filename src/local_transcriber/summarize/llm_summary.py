"""LLM-based summarization for transcriptions."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_summary(
    transcript: str,
    model: str = "gemini",
    output_path: Path | None = None,
) -> dict[str, Any] | None:
    """
    Genera un resumen de la transcripción usando un LLM.

    Args:
        transcript: Texto de la transcripción completa
        model: Modelo a usar ('gemini' o 'claude')
        output_path: Ruta opcional donde guardar el resumen

    Returns:
        Diccionario con el resumen estructurado o None si falla
    """
    if model.lower() == "gemini":
        return _generate_summary_gemini(transcript, output_path)
    elif model.lower() == "claude":
        return _generate_summary_claude(transcript, output_path)
    else:
        logger.error("Unsupported model: %s. Supported: gemini, claude", model)
        return None


def _generate_summary_gemini(
    transcript: str, output_path: Path | None = None
) -> dict[str, Any] | None:
    """
    Genera resumen usando Google Gemini API.

    Args:
        transcript: Texto de la transcripción
        output_path: Ruta opcional donde guardar el resumen

    Returns:
        Diccionario con el resumen estructurado o None si falla
    """
    try:
        import google.generativeai as genai
    except ImportError:
        logger.error(
            "google-generativeai not installed. Install with: pip install google-generativeai"
        )
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")

        prompt = f"""Analiza esta transcripción de llamada/reunión y genera un resumen estructurado en formato JSON.

Transcripción:
{transcript}

Genera un JSON con la siguiente estructura:
{{
  "summary": "Resumen ejecutivo en 3-5 oraciones",
  "key_points": ["punto clave 1", "punto clave 2", ...],
  "action_items": [
    {{
      "task": "descripción de la tarea",
      "assignee": "responsable si se menciona",
      "due_date": "fecha si se menciona"
    }}
  ],
  "decisions": ["decisión 1", "decisión 2", ...],
  "topics": ["tema 1", "tema 2", ...]
}}

Responde SOLO con el JSON, sin texto adicional."""

        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Limpiar respuesta (puede tener markdown code blocks)
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parsear JSON
        summary_data = json.loads(response_text)

        # Guardar si se especificó output_path
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info("Summary saved to: %s", output_path)

        return summary_data

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini response as JSON: %s", e)
        logger.debug("Response text: %s", response_text[:500])
        return None
    except Exception as e:
        logger.error("Error generating summary with Gemini: %s", e, exc_info=True)
        return None


def _generate_summary_claude(
    transcript: str, output_path: Path | None = None
) -> dict[str, Any] | None:
    """
    Genera resumen usando Anthropic Claude API.

    Args:
        transcript: Texto de la transcripción
        output_path: Ruta opcional donde guardar el resumen

    Returns:
        Diccionario con el resumen estructurado o None si falla
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error(
            "anthropic not installed. Install with: pip install anthropic"
        )
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return None

    try:
        client = Anthropic(api_key=api_key)

        prompt = f"""Analiza esta transcripción de llamada/reunión y genera un resumen estructurado en formato JSON.

Transcripción:
{transcript}

Genera un JSON con la siguiente estructura:
{{
  "summary": "Resumen ejecutivo en 3-5 oraciones",
  "key_points": ["punto clave 1", "punto clave 2", ...],
  "action_items": [
    {{
      "task": "descripción de la tarea",
      "assignee": "responsable si se menciona",
      "due_date": "fecha si se menciona"
    }}
  ],
  "decisions": ["decisión 1", "decisión 2", ...],
  "topics": ["tema 1", "tema 2", ...]
}}

Responde SOLO con el JSON, sin texto adicional."""

        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text.strip()

        # Limpiar respuesta (puede tener markdown code blocks)
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parsear JSON
        summary_data = json.loads(response_text)

        # Guardar si se especificó output_path
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info("Summary saved to: %s", output_path)

        return summary_data

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Claude response as JSON: %s", e)
        logger.debug("Response text: %s", response_text[:500])
        return None
    except Exception as e:
        logger.error("Error generating summary with Claude: %s", e, exc_info=True)
        return None
