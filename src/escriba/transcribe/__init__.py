"""Transcription backends and export helpers.

``StreamingTranscriber`` / ``get_device_config`` are resolved lazily (PEP 562)
so that importing lightweight members like ``escriba.transcribe.config`` does
not pull in ``streaming`` — and with it faster-whisper/ctranslate2/transformers
(~1 s of imports) — at app startup.
"""

from typing import TYPE_CHECKING, Any

from escriba.transcribe.config import VADConfig
from escriba.transcribe.formats import export_to_json, export_to_txt

if TYPE_CHECKING:
    from escriba.transcribe.streaming import StreamingTranscriber, get_device_config

__all__ = [
    "StreamingTranscriber",
    "get_device_config",
    "VADConfig",
    "export_to_json",
    "export_to_txt",
]

_LAZY = {"StreamingTranscriber", "get_device_config"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from escriba.transcribe import streaming

        value = getattr(streaming, name)
        globals()[name] = value  # cache so later lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
