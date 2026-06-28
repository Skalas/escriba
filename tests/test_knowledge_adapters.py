"""Tests T54.x — Knowledge store adapters."""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from escriba.config import AppConfig, ConfigValidationError


# T54.1 — Default config has local-markdown provider
def test_default_config_has_local_markdown_provider() -> None:
    """AppConfig default knowledge_store.provider is 'local-markdown'."""
    cfg = AppConfig()
    assert cfg.knowledge_store.provider == "local-markdown"


# T54.2 — LocalMarkdownAdapter writes .md file
def test_local_markdown_adapter_writes_file(tmp_path: Path) -> None:
    """LocalMarkdownAdapter.export() writes a Markdown file to output_dir."""
    from escriba.knowledge.local_markdown import LocalMarkdownAdapter

    session = {
        "id": "abc12345",
        "name": "My Test Session",
        "started_at": "2024-01-01T10:00:00+00:00",
        "duration_seconds": 60.0,
        "notes_text": None,
        "audio_path": None,
    }
    segments = [
        {"id": 1, "start_time": 0.0, "end_time": 5.0, "text": "Hello world", "speaker": None, "speaker_display": None},
    ]
    adapter = LocalMarkdownAdapter(output_dir=str(tmp_path))
    adapter.export(session=session, summary_json=None, audio_path=None, segments=segments)

    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1, f"Expected 1 .md file, got {md_files}"
    content = md_files[0].read_text(encoding="utf-8")
    assert "My Test Session" in content
    assert "Hello world" in content


# T54.3 — Unknown provider raises ConfigValidationError
def test_unknown_knowledge_provider_raises_error() -> None:
    """AppConfig.validate() raises ConfigValidationError for unknown provider."""
    from escriba.config import KnowledgeStoreConfig

    bad_ks = KnowledgeStoreConfig(provider="unknown-provider")
    cfg = AppConfig(knowledge_store=bad_ks)
    with pytest.raises(ConfigValidationError, match="knowledge_store.provider"):
        cfg.validate()


# T54.4 — Export to unwritable dir logs error and doesn't raise
def test_export_to_unwritable_dir_logs_and_does_not_raise(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """LocalMarkdownAdapter.export() swallows OSError and logs it."""
    import logging
    from escriba.knowledge.local_markdown import LocalMarkdownAdapter

    # Create a dir and remove write permission
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    locked_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # read+execute only, no write

    session = {
        "id": "xyz99999",
        "name": "Session",
        "started_at": "2024-01-01T10:00:00+00:00",
        "duration_seconds": 10.0,
        "notes_text": None,
        "audio_path": None,
    }
    adapter = LocalMarkdownAdapter(output_dir=str(locked_dir))
    with caplog.at_level(logging.ERROR, logger="escriba.knowledge.local_markdown"):
        try:
            # Should not raise even on permission error
            adapter.export(session=session, summary_json=None, audio_path=None, segments=[])
        except Exception as exc:
            pytest.fail(f"export() raised unexpectedly: {exc}")
        finally:
            # Restore permissions to allow cleanup
            locked_dir.chmod(stat.S_IRWXU)

    assert any(
        "export failed" in r.message.lower()
        for r in caplog.records
    ), "Expected an error to be logged when export to unwritable dir fails"
