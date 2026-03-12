"""GitHub integration for creating issues from transcriptions."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_action_items(transcript: str, model: str = "gemini") -> list[dict[str, Any]]:
    """
    Extrae action items de una transcripción usando LLM.

    Args:
        transcript: Texto de la transcripción
        model: Modelo LLM a usar ('gemini' o 'claude')

    Returns:
        Lista de action items con 'task', 'assignee', 'due_date'
    """
    try:
        from local_transcriber.summarize import generate_summary

        summary = generate_summary(transcript, model=model)
        if summary and "action_items" in summary:
            return summary["action_items"]
        return []

    except Exception as e:
        logger.error("Error extracting action items: %s", e, exc_info=True)
        return []


def create_github_issue(
    repo: str,
    title: str,
    body: str,
    labels: list[str] | None = None,
    token: str | None = None,
) -> dict[str, Any] | None:
    """
    Crea un issue en GitHub.

    Args:
        repo: Repositorio en formato 'owner/repo'
        title: Título del issue
        body: Cuerpo del issue
        labels: Lista de labels (opcional)
        token: GitHub token (o usar GITHUB_TOKEN env var)

    Returns:
        Diccionario con información del issue creado o None si falla
    """
    token = token or os.getenv("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN environment variable not set")
        return None

    try:
        from github import Github

        g = Github(token)
        repository = g.get_repo(repo)

        issue = repository.create_issue(
            title=title,
            body=body,
            labels=labels or [],
        )

        logger.info("Created GitHub issue: %s", issue.html_url)
        return {
            "number": issue.number,
            "title": issue.title,
            "url": issue.html_url,
        }

    except ImportError:
        logger.error("PyGithub not installed. Install with: pip install PyGithub")
        return None
    except Exception as e:
        logger.error("Error creating GitHub issue: %s", e, exc_info=True)
        return None


def create_issues_from_transcript(
    transcript_path: Path,
    repo: str,
    model: str = "gemini",
    token: str | None = None,
) -> list[dict[str, Any]]:
    """
    Crea issues de GitHub desde una transcripción.

    Args:
        transcript_path: Ruta al archivo de transcripción
        repo: Repositorio en formato 'owner/repo'
        model: Modelo LLM para extraer action items
        token: GitHub token (opcional)

    Returns:
        Lista de issues creados
    """
    # Leer transcripción
    try:
        with transcript_path.open("r", encoding="utf-8") as f:
            transcript = f.read()
    except Exception as e:
        logger.error("Error reading transcript: %s", e, exc_info=True)
        return []

    # Extraer action items
    action_items = extract_action_items(transcript, model=model)

    if not action_items:
        logger.info("No action items found in transcript")
        return []

    # Crear issues
    created_issues = []
    for item in action_items:
        task = item.get("task", "")
        assignee = item.get("assignee", "")
        due_date = item.get("due_date", "")

        if not task:
            continue

        # Crear título
        title = task[:100]  # Limitar longitud

        # Crear cuerpo
        body_parts = [f"**Tarea:** {task}"]
        if assignee:
            body_parts.append(f"**Asignado a:** {assignee}")
        if due_date:
            body_parts.append(f"**Fecha límite:** {due_date}")
        body_parts.append(f"\n**Contexto:**\n```\n{transcript[:500]}...\n```")

        body = "\n\n".join(body_parts)

        # Crear issue
        issue = create_github_issue(
            repo=repo,
            title=title,
            body=body,
            labels=["transcription", "action-item"],
            token=token,
        )

        if issue:
            created_issues.append(issue)

    return created_issues
