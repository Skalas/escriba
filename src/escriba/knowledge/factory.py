"""Knowledge-store provider factory."""
from __future__ import annotations

from typing import TYPE_CHECKING

from escriba.knowledge.port import KnowledgeStore

if TYPE_CHECKING:
    from escriba.config import KnowledgeStoreConfig


def get_knowledge_store(ks_config: "KnowledgeStoreConfig") -> KnowledgeStore:
    """Return the KnowledgeStore implementation for the configured provider."""
    from typing import Callable

    from escriba.knowledge.local_markdown import LocalMarkdownAdapter

    _registry: dict[str, Callable[[], KnowledgeStore]] = {
        "local-markdown": lambda: LocalMarkdownAdapter(
            output_dir=ks_config.local_markdown.output_dir
        ),
    }
    factory = _registry.get(ks_config.provider)
    if factory is None:
        raise ValueError(f"Unknown knowledge provider: {ks_config.provider!r}")
    return factory()
