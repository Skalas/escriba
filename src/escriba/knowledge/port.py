"""KnowledgeStore port — abstract export interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class KnowledgeStore(ABC):
    """Port for exporting session data to external knowledge stores."""

    @abstractmethod
    def export(
        self,
        session: dict[str, Any],
        summary_json: dict[str, Any] | None,
        audio_path: Path | None,
        segments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Export a completed session. Must not raise — log failures instead."""
