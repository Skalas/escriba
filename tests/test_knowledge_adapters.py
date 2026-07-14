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


def test_webhook_adapter_posts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebhookAdapter POSTs session JSON and reads auth token from env."""
    from escriba.knowledge.webhook import WebhookAdapter

    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request, timeout=30):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("ESCRIBA_WEBHOOK_TOKEN", "secret-token")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    session = {
        "id": "abc12345",
        "name": "Webhook Session",
        "started_at": "2024-01-01T10:00:00+00:00",
        "duration_seconds": 10.0,
        "notes_text": None,
        "audio_path": None,
    }
    adapter = WebhookAdapter(url="https://example.com/hook")
    adapter.export(session=session, summary_json=None, audio_path=None, segments=[])

    assert captured["url"] == "https://example.com/hook"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers.get("Authorization") == "Bearer secret-token"
    body = captured["body"]
    assert isinstance(body, bytes)
    assert b"Webhook Session" in body
    assert captured["timeout"] == 10.0


def test_webhook_adapter_rejects_disallowed_auth_env() -> None:
    from escriba.knowledge.url_safety import WebhookUrlError, validate_webhook_auth_env

    with pytest.raises(WebhookUrlError, match="auth_env"):
        validate_webhook_auth_env("GEMINI_API_KEY")


def test_webhook_adapter_rejects_http_url() -> None:
    from escriba.knowledge.webhook import WebhookAdapter

    with pytest.raises(Exception, match="https"):
        WebhookAdapter(url="http://example.com/hook")


def test_webhook_adapter_rejects_private_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    from escriba.knowledge.webhook import WebhookAdapter

    monkeypatch.setattr(
        "escriba.knowledge.url_safety.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("192.168.1.1", 0))],
    )
    with pytest.raises(Exception, match="private"):
        WebhookAdapter(url="https://internal.corp.example/hook")


def test_webhook_provider_rejects_bad_auth_env() -> None:
    from escriba.config import KnowledgeStoreConfig, KnowledgeStoreWebhookConfig

    cfg = AppConfig(
        knowledge_store=KnowledgeStoreConfig(
            provider="webhook",
            webhook=KnowledgeStoreWebhookConfig(
                url="https://example.com/hook",
                auth_env="ANTHROPIC_API_KEY",
            ),
        )
    )
    with pytest.raises(ConfigValidationError, match="auth_env"):
        cfg.validate()


def test_webhook_adapter_swallows_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebhookAdapter must not raise on HTTP failures."""
    import urllib.error

    from escriba.knowledge.webhook import WebhookAdapter

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise urllib.error.HTTPError("https://x", 500, "fail", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    adapter = WebhookAdapter(url="https://example.com/hook")
    adapter.export(
        session={"id": "x", "name": "n"},
        summary_json=None,
        audio_path=None,
        segments=[],
    )


def test_custom_script_adapter_invokes_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CustomScriptAdapter runs the script with JSON on stdin (no shell)."""
    from escriba.knowledge.custom_script import CustomScriptAdapter

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "export.sh"
    script.write_text("#!/bin/sh\ncat\n", encoding="utf-8")
    script.chmod(0o755)

    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    session = {"id": "script1", "name": "Script Session"}
    adapter = CustomScriptAdapter(
        script_path="export.sh",
        scripts_dir=str(scripts_dir),
        timeout_seconds=5.0,
    )
    adapter.export(session=session, summary_json=None, audio_path=None, segments=[])

    assert captured["cmd"] == [str(script)]
    stdin = captured["input"]
    assert isinstance(stdin, str)
    assert "Script Session" in stdin


def test_custom_script_rejects_path_outside_scripts_dir(tmp_path: Path) -> None:
    from escriba.knowledge.custom_script import CustomScriptAdapter

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    outside = tmp_path / "evil.sh"
    outside.write_text("#!/bin/sh\n", encoding="utf-8")
    outside.chmod(0o755)

    adapter = CustomScriptAdapter(
        script_path=str(outside),
        scripts_dir=str(scripts_dir),
    )
    adapter.export(
        session={"id": "x", "name": "n"},
        summary_json=None,
        audio_path=None,
        segments=[],
    )


def test_custom_script_adapter_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CustomScriptAdapter logs and returns on timeout without raising."""
    import subprocess

    from escriba.knowledge.custom_script import CustomScriptAdapter

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "slow.sh"
    script.write_text("#!/bin/sh\nsleep 2\n", encoding="utf-8")
    script.chmod(0o755)

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr("subprocess.run", fake_run)
    adapter = CustomScriptAdapter(
        script_path="slow.sh",
        scripts_dir=str(scripts_dir),
        timeout_seconds=0.01,
    )
    adapter.export(
        session={"id": "slow", "name": "Slow"},
        summary_json=None,
        audio_path=None,
        segments=[],
    )


def test_webhook_provider_validates_url() -> None:
    from escriba.config import KnowledgeStoreConfig, KnowledgeStoreWebhookConfig

    cfg = AppConfig(
        knowledge_store=KnowledgeStoreConfig(
            provider="webhook",
            webhook=KnowledgeStoreWebhookConfig(url=""),
        )
    )
    with pytest.raises(ConfigValidationError, match="webhook.url"):
        cfg.validate()


def test_knowledge_export_timeout_capped_in_config() -> None:
    from escriba.config import KnowledgeStoreConfig, KnowledgeStoreWebhookConfig
    from escriba.knowledge.constants import EXPORT_TIMEOUT_CAP_SECONDS

    cfg = AppConfig(
        knowledge_store=KnowledgeStoreConfig(
            provider="webhook",
            webhook=KnowledgeStoreWebhookConfig(
                url="https://example.com/hook",
                timeout_seconds=EXPORT_TIMEOUT_CAP_SECONDS + 1,
            ),
        )
    )
    with pytest.raises(ConfigValidationError, match="timeout_seconds"):
        cfg.validate()
