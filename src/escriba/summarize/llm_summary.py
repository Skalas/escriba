"""LLM-based summarization for transcriptions."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6-20250514"


def resolve_provider_and_model(model: str) -> tuple[str, str]:
    """Resolve a model string to (provider, model_id).

    Accepts either a provider shorthand ("gemini", "claude") which maps to
    the default model, or a full model ID like "gemini-2.5-flash-preview"
    or "claude-sonnet-4-6-20250514".
    """
    m = model.lower().strip()
    if m == "gemini":
        return "gemini", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if m == "claude":
        return "claude", os.getenv("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL)
    if m.startswith("gemini"):
        return "gemini", m
    if m.startswith("claude"):
        return "claude", m
    # Unknown — try gemini as fallback
    logger.warning("Unknown model prefix %r, assuming gemini", model)
    return "gemini", m


def generate_summary(
    transcript: str,
    model: str = "gemini",
    output_path: Path | None = None,
) -> dict[str, Any] | None:
    """
    Genera un resumen de la transcripción usando un LLM.

    Accepts a provider name ("gemini"/"claude") or a full model ID.
    """
    provider, model_id = resolve_provider_and_model(model)
    if provider == "gemini":
        return _generate_summary_gemini(transcript, output_path, model_id=model_id)
    elif provider == "claude":
        return _generate_summary_claude(transcript, output_path, model_id=model_id)
    else:
        logger.error("Unsupported provider: %s", provider)
        return None


def _generate_summary_gemini(
    transcript: str, output_path: Path | None = None, *, model_id: str = DEFAULT_GEMINI_MODEL
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
        from google import genai
    except ImportError:
        logger.error(
            "google-genai not installed. Install with: pip install google-genai"
        )
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None

    try:
        client = genai.Client(api_key=api_key)

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

        response = client.models.generate_content(model=model_id, contents=prompt)
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
    transcript: str, output_path: Path | None = None, *, model_id: str = DEFAULT_CLAUDE_MODEL
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
            model=model_id,
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


def list_available_models() -> dict[str, list[str]]:
    """Fetch available model IDs from configured providers.

    Returns a dict like {"gemini": [...], "claude": [...]}.
    Only queries providers whose API key is set.
    """
    result: dict[str, list[str]] = {}

    if os.getenv("GEMINI_API_KEY", "").strip():
        result["gemini"] = _list_gemini_models()

    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        result["claude"] = _list_claude_models()

    return result


def _list_gemini_models() -> list[str]:
    try:
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        models = []
        for model in client.models.list():
            name = model.name
            # API returns "models/gemini-..." — strip the prefix
            if name.startswith("models/"):
                name = name[7:]
            # Only include generative models (skip embedding, etc.)
            if "gemini" in name:
                models.append(name)
        return sorted(models) if models else [DEFAULT_GEMINI_MODEL]
    except ImportError:
        logger.warning("google-genai SDK not installed")
        return [DEFAULT_GEMINI_MODEL]
    except Exception as e:
        logger.error("Failed to list Gemini models: %s", e)
        return [DEFAULT_GEMINI_MODEL]


def _list_claude_models() -> list[str]:
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # models.list() requires anthropic >= 0.39.0
        if not hasattr(client, "models"):
            return [DEFAULT_CLAUDE_MODEL]
        response = client.models.list(limit=100)
        models = [m.id for m in response.data if m.id.startswith("claude")]
        return sorted(models) if models else [DEFAULT_CLAUDE_MODEL]
    except ImportError:
        logger.warning("anthropic SDK not installed")
        return [DEFAULT_CLAUDE_MODEL]
    except Exception as e:
        logger.error("Failed to list Claude models: %s", e)
        return [DEFAULT_CLAUDE_MODEL]
