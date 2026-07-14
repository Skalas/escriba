"""Tests for install path inventory and upgrade step alignment."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from escriba.app.install_paths import (
    INSTALL_STEPS,
    UPGRADE_PROGRESS_STEP_IDS,
    install_path_inventory,
)
from escriba.app.updates import _execute_upgrade


def test_install_path_inventory_lists_all_steps() -> None:
    rows = install_path_inventory()
    assert len(rows) == len(INSTALL_STEPS)
    assert {row["id"] for row in rows} == {step.id for step in INSTALL_STEPS}


def test_upgrade_progress_ids_derived_from_inventory() -> None:
    expected = tuple(step.id for step in INSTALL_STEPS if step.in_app_upgrade)
    assert UPGRADE_PROGRESS_STEP_IDS == expected
    inventory_ids = {step.id for step in INSTALL_STEPS}
    assert set(UPGRADE_PROGRESS_STEP_IDS).issubset(inventory_ids)


def test_precheck_macos_not_in_upgrade_progress() -> None:
    assert "precheck_macos" not in UPGRADE_PROGRESS_STEP_IDS
    by_id = {step.id: step for step in INSTALL_STEPS}
    assert by_id["precheck_macos"].in_app_upgrade is False


def test_in_app_upgrade_includes_core_build_steps() -> None:
    by_id = {step.id: step for step in INSTALL_STEPS}
    for step_id in ("uv_sync", "swift_binary", "build_app", "install_app"):
        assert by_id[step_id].in_app_upgrade is True


def test_make_install_is_minimal_dev_path() -> None:
    by_id = {step.id: step for step in INSTALL_STEPS}
    assert by_id["build_app"].make_install is True
    assert by_id["install_app"].make_install is True
    assert by_id["git_fetch"].make_install is False
    assert by_id["uv_sync"].make_install is False


def test_execute_upgrade_reports_inventory_step_ids(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    (project / ".git").mkdir()
    dist_app = project / "dist" / "Escriba.app"
    dist_app.mkdir(parents=True)
    progress_steps: list[str] = []

    def on_progress(step: str, message: str) -> None:
        progress_steps.append(step)

    with (
        patch("escriba.app.updates._run_git", return_value=type("R", (), {"returncode": 0, "stderr": ""})()),
        patch("escriba.app.updates._fast_forward_to_release"),
        patch("escriba.app.updates.subprocess.run", return_value=type("R", (), {"returncode": 0, "stderr": ""})()),
        patch("escriba.app.updates._ensure_swift_binary"),
        patch("escriba.app.updates._install_app_bundle"),
    ):
        _execute_upgrade(project, release_tag="v1.3.1", assets=(), on_progress=on_progress)

    assert set(progress_steps) == set(UPGRADE_PROGRESS_STEP_IDS)
    assert progress_steps == list(UPGRADE_PROGRESS_STEP_IDS)
