"""Tests for GitHub update checks and upgrade service."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from escriba.app.server import AppState
from escriba.app.updates import (
    UpgradeAlreadyInProgress,
    UpgradePreflightError,
    UpgradeService,
    UpgradeStatus,
    _download_bytes,
    _ensure_swift_binary,
    _safe_extract_tar,
    _validate_asset_url,
    check_for_updates,
    compare_versions,
    run_upgrade_blocking,
)
from escriba.config import AppConfig
from tests.conftest import make_handler


@pytest.fixture
def minimal_config(tmp_path: Path) -> AppConfig:
    cfg_path = tmp_path / "escriba.toml"
    cfg_path.write_text(
        """
[audio]
audio_source = "mic"
sample_rate = 16000
channels = 1

[streaming]
backend = "mlx-whisper"
model_size = "tiny"
chunk_duration = 0.5
""".strip(),
        encoding="utf-8",
    )
    return AppConfig.load(cfg_path)


@pytest.fixture
def app_state(minimal_config: AppConfig, tmp_path: Path) -> AppState:
    from escriba.app.database import Database

    return AppState(config=minimal_config, db=Database(tmp_path / "updates.db"))


def _release_payload(tag: str = "v1.4.0", body: str = "Bug fixes") -> bytes:
    return json.dumps(
        {
            "tag_name": tag,
            "html_url": f"https://github.com/Skalas/escriba/releases/tag/{tag}",
            "body": body,
            "assets": [
                {
                    "name": "audio-capture-arm64-darwin.tar.gz",
                    "browser_download_url": (
                        "https://github.com/Skalas/escriba/releases/download/"
                        f"{tag}/audio-capture-arm64-darwin.tar.gz"
                    ),
                }
            ],
        }
    ).encode("utf-8")


def test_compare_versions_semver_order() -> None:
    assert compare_versions("1.3.0", "1.4.0") < 0
    assert compare_versions("1.10.0", "1.9.0") > 0
    assert compare_versions("v1.3.0", "1.3.0") == 0


def test_check_for_updates_detects_newer_release() -> None:
    def opener(request, timeout=15):  # noqa: ARG001
        return io.BytesIO(_release_payload())

    result = check_for_updates(current_version="1.3.0", opener=opener)
    assert result.ok is True
    assert result.update_available is True
    assert result.latest == "v1.4.0"
    assert result.release_url is not None
    assert result.assets


def test_check_for_updates_skips_disallowed_asset_urls() -> None:
    payload = json.dumps(
        {
            "tag_name": "v9.9.9",
            "html_url": "https://github.com/Skalas/escriba/releases/tag/v9.9.9",
            "body": "",
            "assets": [
                {
                    "name": "audio-capture-arm64-darwin.tar.gz",
                    "browser_download_url": "https://evil.example/asset.tar.gz",
                }
            ],
        }
    ).encode()

    def opener(request, timeout=15):  # noqa: ARG001
        return io.BytesIO(payload)

    result = check_for_updates(current_version="1.0.0", opener=opener)
    assert result.assets == ()


def test_check_for_updates_fail_soft_on_network_error() -> None:
    import urllib.error

    def opener(request, timeout=15):  # noqa: ARG001
        raise urllib.error.URLError("offline")

    result = check_for_updates(current_version="1.3.0", opener=opener)
    assert result.ok is True
    assert result.update_available is False
    assert result.error == "Could not reach GitHub"


def test_validate_asset_url_rejects_off_allowlist() -> None:
    with pytest.raises(ValueError):
        _validate_asset_url("https://evil.example/asset.tar.gz")


def test_validate_asset_url_rejects_wrong_github_path() -> None:
    with pytest.raises(ValueError):
        _validate_asset_url(
            "https://github.com/Skalas/escriba/raw/main/evil.tar.gz",
            owner="Skalas",
            repo="escriba",
        )


def test_validate_asset_url_accepts_release_download_path() -> None:
    url = (
        "https://github.com/Skalas/escriba/releases/download/v1.4.0/"
        "audio-capture-arm64-darwin.tar.gz"
    )
    assert _validate_asset_url(url, owner="Skalas", repo="escriba") == url


def test_download_bytes_rejects_off_allowlist() -> None:
    with pytest.raises(ValueError):
        _download_bytes("https://evil.example/asset.tar.gz")


def test_upgrade_status_idle_serializes_ok_null() -> None:
    payload = UpgradeStatus().to_dict()
    assert payload["ok"] is None
    assert payload["completed"] is False


def test_get_update_check_handler_caches_result(app_state: AppState) -> None:
    handler = make_handler(app_state)
    fake = check_for_updates(current_version="1.3.0", opener=lambda *_a, **_k: io.BytesIO(_release_payload()))
    with patch("escriba.app.server.check_for_updates", return_value=fake) as mocked:
        payload, status = handler._get_update_check()
    assert status == 200
    assert payload["update_available"] is True
    assert app_state.last_update_check is not None
    mocked.assert_called_once()


def test_start_update_install_rejects_when_no_update(app_state: AppState) -> None:
    handler = make_handler(app_state)
    with patch(
        "escriba.app.server.check_for_updates",
        return_value=check_for_updates(current_version="9.9.9", opener=lambda *_a, **_k: io.BytesIO(_release_payload("v1.3.0"))),
    ):
        payload, status = handler._start_update_install()
    assert status == 409
    assert payload["ok"] is False


def test_upgrade_service_rejects_dirty_tree(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    (project / ".git").mkdir()
    service = UpgradeService()
    with (
        patch("escriba.app.updates.resolve_project_dir", return_value=project),
        patch(
            "escriba.app.updates.git_worktree_state",
            return_value={"dirty": True, "branch": "main"},
        ),
        patch("escriba.app.updates.platform.system", return_value="Darwin"),
    ):
        with pytest.raises(UpgradePreflightError):
            service.try_begin()


def test_upgrade_service_rejects_concurrent_run() -> None:
    service = UpgradeService()
    service._claim()  # noqa: SLF001
    with pytest.raises(UpgradeAlreadyInProgress):
        service.try_begin()
    service._abort_claim()  # noqa: SLF001


def test_safe_extract_tar_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="../escape")
        info.size = 0
        tar.addfile(info)

    with pytest.raises(RuntimeError, match="Unsafe archive path"):
        _safe_extract_tar(archive, tmp_path / "dest")


def test_ensure_swift_binary_always_downloads_when_asset_present(
    tmp_path: Path,
) -> None:
    project = tmp_path / "repo"
    swift_bin_dir = project / "swift-audio-capture" / ".build" / "release"
    swift_bin_dir.mkdir(parents=True)
    existing = swift_bin_dir / "audio-capture"
    existing.write_text("old", encoding="utf-8")
    existing.chmod(0o755)

    assets = (
        {
            "name": "audio-capture-arm64-darwin.tar.gz",
            "url": "https://github.com/Skalas/escriba/releases/download/v1.4.0/audio-capture-arm64-darwin.tar.gz",
        },
    )

    with patch("escriba.app.updates._download_bytes", return_value=b"tar") as download:
        with patch("escriba.app.updates._safe_extract_tar") as extract:
            _ensure_swift_binary(project, assets)

    download.assert_called_once()
    extract.assert_called_once()


def test_run_upgrade_blocking_invokes_steps(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    (project / ".git").mkdir()

    with (
        patch("escriba.app.updates.resolve_project_dir", return_value=project),
        patch(
            "escriba.app.updates.git_worktree_state",
            return_value={"dirty": False, "branch": "main"},
        ),
        patch("escriba.app.updates.platform.system", return_value="Darwin"),
        patch("escriba.app.updates._execute_upgrade", return_value={"ok": True}) as execute,
    ):
        result = run_upgrade_blocking(release_tag="v1.4.0", assets=())

    assert result["ok"] is True
    execute.assert_called_once()
