"""GitHub release update checks and guarded in-app upgrade runner."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tarfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, cast

from escriba import __version__
from escriba.app.install_paths import upgrade_progress_message

logger = logging.getLogger(__name__)

DEFAULT_GITHUB_OWNER = "Skalas"
DEFAULT_GITHUB_REPO = "escriba"
GITHUB_SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
RELEASE_TAG_RE = re.compile(r"^v?\d+\.\d+\.\d+$")
RELEASES_LATEST_URL = "https://api.github.com/repos/{owner}/{repo}/releases/latest"
NOTES_SNIPPET_MAX_CHARS = 400
SWIFT_ASSET_SUFFIX = "audio-capture-arm64-darwin.tar.gz"
APP_NAME = "Escriba"
ALLOWED_ASSET_HOSTS = frozenset({"github.com", "objects.githubusercontent.com"})
ALLOWED_API_HOSTS = frozenset({"api.github.com"})
USER_UPDATE_FAILED = "Update failed; check logs"

ProgressCallback = Callable[[str, str], None]


class _UrlOpener(Protocol):
    """OpenerDirector-like or test double with ``.open``."""

    def open(self, request: urllib.request.Request, timeout: float | None = None) -> Any:
        """Open a URL request."""


UrlOpener = _UrlOpener | urllib.request.OpenerDirector | Callable[..., Any]


def _get_str_env(name: str, default: str) -> str:
    """Return a non-empty environment variable string."""
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _validated_slug(value: str, label: str) -> str:
    if not GITHUB_SLUG_RE.fullmatch(value):
        raise ValueError(f"{label} must match {GITHUB_SLUG_RE.pattern}")
    return value


def github_owner() -> str:
    """Configured GitHub owner for release lookups."""
    return _validated_slug(_get_str_env("ESCRIBA_GITHUB_OWNER", DEFAULT_GITHUB_OWNER), "ESCRIBA_GITHUB_OWNER")


def github_repo() -> str:
    """Configured GitHub repository for release lookups."""
    return _validated_slug(_get_str_env("ESCRIBA_GITHUB_REPO", DEFAULT_GITHUB_REPO), "ESCRIBA_GITHUB_REPO")


def resolve_project_dir() -> Path | None:
    """Best-effort Escriba source tree (dev checkout or ~/.escriba install)."""
    candidates = [
        Path(__file__).resolve().parents[3],
        Path.home() / ".escriba",
    ]
    for candidate in candidates:
        if (candidate / ".git").is_dir():
            return candidate
    return None


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a semver-ish version into comparable integer tuple."""
    cleaned = version.strip().lstrip("vV")
    parts: list[int] = []
    for piece in cleaned.split("."):
        match = re.match(r"(\d+)", piece)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts or (0,))


def compare_versions(left: str, right: str) -> int:
    """Compare two versions. Returns -1, 0, or 1."""
    left_parts = _parse_version(left)
    right_parts = _parse_version(right)
    width = max(len(left_parts), len(right_parts))
    left_padded = left_parts + (0,) * (width - len(left_parts))
    right_padded = right_parts + (0,) * (width - len(right_parts))
    if left_padded < right_padded:
        return -1
    if left_padded > right_padded:
        return 1
    return 0


def resolve_installed_version() -> str:
    """Return the real installed package version (never soak-overridden)."""
    return __version__


def version_override_active() -> bool:
    """True when ``ESCRIBA_VERSION_OVERRIDE`` is set for check-only soak tests."""
    return bool(os.getenv("ESCRIBA_VERSION_OVERRIDE", "").strip())


def resolve_check_version(*, override: str | None = None) -> str:
    """Return the version used for update-check / About display.

    Precedence: explicit ``override`` (CLI ``--current`` / tests) →
    ``ESCRIBA_VERSION_OVERRIDE`` env → installed ``__version__``.

    Blank ``override`` is treated as absent. Mutating upgrade paths must use
    ``resolve_installed_version()`` instead.
    """
    if override is not None:
        value = override.strip()
        if value:
            return value
    if version_override_active():
        return os.getenv("ESCRIBA_VERSION_OVERRIDE", "").strip()
    return __version__


def _validate_asset_url(
    url: str,
    *,
    owner: str | None = None,
    repo: str | None = None,
) -> str:
    """Reject asset URLs outside the GitHub release download allowlist."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme != "https" or parsed.netloc not in ALLOWED_ASSET_HOSTS:
        raise ValueError(f"Asset URL host not allowed: {parsed.netloc!r}")
    if parsed.netloc == "github.com":
        owner_name = owner or github_owner()
        repo_name = repo or github_repo()
        prefix = f"/{owner_name}/{repo_name}/releases/download/"
        if not parsed.path.startswith(prefix):
            raise ValueError(f"Asset URL path must start with {prefix!r}")
    return url


def _validate_api_url(url: str) -> str:
    """Reject GitHub API URLs outside api.github.com."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme != "https" or parsed.netloc not in ALLOWED_API_HOSTS:
        raise ValueError(f"API URL host not allowed: {parsed.netloc!r}")
    return url


class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects only when the target stays on the asset allowlist."""

    def __init__(self, *, owner: str | None = None, repo: str | None = None) -> None:
        super().__init__()
        self._owner = owner
        self._repo = repo

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_asset_url(newurl, owner=self._owner, repo=self._repo)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _ApiRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects only when the target stays on api.github.com."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_api_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _asset_download_opener(
    *,
    owner: str | None = None,
    repo: str | None = None,
) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        _AllowlistedRedirectHandler(owner=owner, repo=repo)
    )


def _api_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_ApiRedirectHandler())


def _open_request(
    opener: UrlOpener,
    request: urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    """Open via OpenerDirector or a bare callable (tests)."""
    open_fn = opener.open if hasattr(opener, "open") else cast(Callable[..., Any], opener)
    return open_fn(request, timeout=timeout)


def _download_bytes(
    url: str,
    *,
    opener: UrlOpener | None = None,
    owner: str | None = None,
    repo: str | None = None,
) -> bytes:
    """Download bytes from an allowlisted GitHub asset URL."""
    safe_url = _validate_asset_url(url, owner=owner, repo=repo)
    request = urllib.request.Request(
        safe_url,
        headers={"User-Agent": "escriba-updater"},
    )
    active: UrlOpener = (
        opener if opener is not None else _asset_download_opener(owner=owner, repo=repo)
    )
    with _open_request(active, request, timeout=120) as response:
        final_url = getattr(response, "url", safe_url)
        _validate_asset_url(final_url, owner=owner, repo=repo)
        return response.read()


@dataclass(frozen=True)
class UpdateCheckResult:
    """Structured update-check response."""

    ok: bool
    current: str
    latest: str | None = None
    update_available: bool = False
    release_url: str | None = None
    release_notes_snippet: str | None = None
    assets: tuple[dict[str, str], ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable payload."""
        return {
            "ok": self.ok,
            "current": self.current,
            "latest": self.latest,
            "update_available": self.update_available,
            "release_url": self.release_url,
            "release_notes_snippet": self.release_notes_snippet,
            "assets": list(self.assets),
            "error": self.error,
        }


def _snippet_from_notes(body: str | None) -> str | None:
    if not body:
        return None
    text = body.strip()
    if len(text) <= NOTES_SNIPPET_MAX_CHARS:
        return text
    return text[: NOTES_SNIPPET_MAX_CHARS - 1].rstrip() + "…"


def _swift_asset_url(assets: tuple[dict[str, str], ...]) -> str | None:
    for asset in assets:
        name = asset.get("name", "")
        if name.endswith(SWIFT_ASSET_SUFFIX) or SWIFT_ASSET_SUFFIX in name:
            url = asset.get("url", "")
            if url:
                return url
    return None


def _fetch_latest_release(
    owner: str | None = None,
    repo: str | None = None,
    *,
    opener: UrlOpener | None = None,
) -> dict[str, Any]:
    """Fetch GitHub ``releases/latest`` JSON."""
    owner_name = owner or github_owner()
    repo_name = repo or github_repo()
    url = RELEASES_LATEST_URL.format(owner=owner_name, repo=repo_name)
    _validate_api_url(url)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "escriba-update-check",
        },
    )
    active: UrlOpener = opener if opener is not None else _api_opener()
    with _open_request(active, request, timeout=15) as response:
        final_url = getattr(response, "url", url)
        _validate_api_url(final_url)
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def check_for_updates(
    *,
    override: str | None = None,
    soak: bool = True,
    owner: str | None = None,
    repo: str | None = None,
    opener: UrlOpener | None = None,
) -> UpdateCheckResult:
    """Compare the running version to GitHub ``releases/latest``.

    ``soak=False`` ignores ``ESCRIBA_VERSION_OVERRIDE`` and uses the installed
    version (unless ``override`` is set). Use that for mutating upgrade paths.

    Fail-soft: network and parse errors return ``ok=True`` with
    ``update_available=False`` and an ``error`` hint instead of raising.
    """
    if soak:
        current = resolve_check_version(override=override)
    else:
        current = override.strip() if override and override.strip() else resolve_installed_version()
    try:
        owner_name = owner or github_owner()
        repo_name = repo or github_repo()
        release = _fetch_latest_release(owner_name, repo_name, opener=opener)
    except urllib.error.URLError as exc:
        logger.info("Update check failed (network): %s", exc)
        return UpdateCheckResult(
            ok=True,
            current=current,
            update_available=False,
            error="Could not reach GitHub",
        )
    except (TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
        logger.info("Update check failed: %s", exc)
        return UpdateCheckResult(
            ok=True,
            current=current,
            update_available=False,
            error="Could not parse release metadata",
        )

    latest = str(release.get("tag_name") or release.get("name") or "").strip() or None
    if not latest:
        return UpdateCheckResult(
            ok=True,
            current=current,
            update_available=False,
            error="Release has no version tag",
        )

    assets: list[dict[str, str]] = []
    for asset in release.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "").strip()
        url = str(asset.get("browser_download_url") or "").strip()
        if not name or not url:
            continue
        try:
            _validate_asset_url(url, owner=owner_name, repo=repo_name)
        except ValueError:
            logger.warning("Skipping release asset with disallowed URL: %s", name)
            continue
        assets.append({"name": name, "url": url})

    release_url = str(release.get("html_url") or "").strip() or None
    notes = _snippet_from_notes(str(release.get("body") or ""))
    newer = compare_versions(current, latest) < 0
    return UpdateCheckResult(
        ok=True,
        current=current,
        latest=latest,
        update_available=newer,
        release_url=release_url,
        release_notes_snippet=notes,
        assets=tuple(assets),
    )


def _run_git(project_dir: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def git_worktree_state(project_dir: Path) -> dict[str, Any]:
    """Return porcelain status and whether a fast-forward pull is plausible."""
    status = _run_git(project_dir, "status", "--porcelain", timeout=10)
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else True
    branch = _run_git(project_dir, "rev-parse", "--abbrev-ref", "HEAD", timeout=10)
    current_branch = branch.stdout.strip() if branch.returncode == 0 else None
    return {
        "dirty": dirty,
        "branch": current_branch,
        "project_dir": str(project_dir),
    }


def _validate_release_tag(release_tag: str) -> str:
    tag = release_tag.strip()
    if not RELEASE_TAG_RE.fullmatch(tag):
        raise RuntimeError("Invalid release tag")
    return tag


def _default_branch(project_dir: Path) -> str:
    sym = _run_git(project_dir, "symbolic-ref", "refs/remotes/origin/HEAD", "--short")
    if sym.returncode == 0:
        ref = sym.stdout.strip()
        if ref.startswith("origin/"):
            return ref[len("origin/") :]
    for candidate in ("main", "master"):
        chk = _run_git(project_dir, "rev-parse", "--verify", candidate)
        if chk.returncode == 0:
            return candidate
    return "main"


def _fast_forward_to_release(project_dir: Path, release_tag: str | None) -> None:
    """Move the tracking branch to a release tag without detached HEAD."""
    if not release_tag:
        pull = _run_git(project_dir, "pull", "--ff-only")
        if pull.returncode != 0:
            raise RuntimeError(pull.stderr.strip() or "git pull --ff-only failed")
        return

    tag = _validate_release_tag(release_tag)
    branch = _default_branch(project_dir)
    checkout = _run_git(project_dir, "checkout", branch)
    if checkout.returncode != 0:
        raise RuntimeError(checkout.stderr.strip() or f"git checkout {branch} failed")
    merge = _run_git(project_dir, "merge", "--ff-only", tag)
    if merge.returncode != 0:
        raise RuntimeError(merge.stderr.strip() or f"git merge --ff-only {tag} failed")


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    """Extract a tar.gz only when every member stays under dest_dir."""
    dest_resolved = dest_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError("Unexpected archive entry")
            target = (dest_dir / member.name).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise RuntimeError("Unsafe archive path")
        extract_kwargs: dict[str, Any] = {}
        if hasattr(tarfile, "data_filter"):
            extract_kwargs["filter"] = "data"
        tar.extractall(dest_dir, **extract_kwargs)


def _ensure_swift_binary(
    project_dir: Path,
    assets: tuple[dict[str, str], ...],
    *,
    opener: UrlOpener | None = None,
    owner: str | None = None,
    repo: str | None = None,
) -> None:
    """Download and extract the release Swift audio-capture binary."""
    asset_url = _swift_asset_url(assets)
    if not asset_url:
        raise RuntimeError("audio-capture release asset not found")

    swift_bin_dir = project_dir / "swift-audio-capture" / ".build" / "release"
    swift_bin = swift_bin_dir / "audio-capture"
    swift_bin_dir.mkdir(parents=True, exist_ok=True)
    archive_path = swift_bin_dir / "audio-capture.tar.gz"
    archive_path.write_bytes(
        _download_bytes(asset_url, opener=opener, owner=owner, repo=repo)
    )
    _safe_extract_tar(archive_path, swift_bin_dir)
    archive_path.unlink(missing_ok=True)
    if not swift_bin.is_file():
        raise RuntimeError("audio-capture binary missing after extract")
    swift_bin.chmod(swift_bin.stat().st_mode | 0o111)
    subprocess.run(
        ["xattr", "-d", "com.apple.quarantine", str(swift_bin)],
        check=False,
        capture_output=True,
    )


def _install_app_bundle(app_src: Path, app_dst: Path) -> None:
    """Atomically replace an .app bundle, keeping the prior bundle on failure."""
    parent = app_dst.parent
    staging = parent / f".{APP_NAME}.app.staging"
    backup = parent / f".{APP_NAME}.app.backup"
    if staging.exists():
        shutil.rmtree(staging)
    if backup.exists():
        shutil.rmtree(backup)
    try:
        shutil.copytree(app_src, staging)
        if app_dst.exists():
            app_dst.rename(backup)
        staging.rename(app_dst)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not app_dst.exists():
            backup.rename(app_dst)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _safe_upgrade_error(exc: BaseException) -> str:
    logger.error("Upgrade failed: %s", exc, exc_info=True)
    return USER_UPDATE_FAILED


def _preflight_upgrade(release_tag: str | None) -> Path:
    """Validate the environment and return the project directory for an upgrade.

    Raises:
        UpgradePreflightError: When preflight checks fail.
    """
    project_dir = resolve_project_dir()
    if project_dir is None:
        raise UpgradePreflightError("Escriba source tree not found")
    if platform.system() != "Darwin":
        raise UpgradePreflightError("Updates are only supported on macOS")
    state = git_worktree_state(project_dir)
    if state["dirty"]:
        raise UpgradePreflightError(
            "Working tree has uncommitted changes — update manually"
        )
    if release_tag:
        _validate_release_tag(release_tag)
    return project_dir


def _execute_upgrade(
    project_dir: Path,
    *,
    release_tag: str | None,
    assets: tuple[dict[str, str], ...],
    on_progress: ProgressCallback | None = None,
    opener: UrlOpener | None = None,
) -> dict[str, Any]:
    """Shared upgrade steps for background and blocking callers."""

    def progress(step: str, message: str | None = None) -> None:
        if on_progress:
            on_progress(step, message or upgrade_progress_message(step))

    progress("git_fetch")
    fetch = _run_git(project_dir, "fetch", "--tags", "origin")
    if fetch.returncode != 0:
        raise RuntimeError(fetch.stderr.strip() or "git fetch failed")

    progress("git_pull")
    _fast_forward_to_release(project_dir, release_tag)

    progress("uv_sync")
    sync = subprocess.run(
        ["uv", "sync"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if sync.returncode != 0:
        raise RuntimeError(sync.stderr.strip() or "uv sync failed")

    progress("swift_binary")
    _ensure_swift_binary(project_dir, assets, opener=opener)

    progress("build_app")
    build = subprocess.run(
        ["uv", "run", "python", "setup_app.py"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if build.returncode != 0:
        raise RuntimeError(build.stderr.strip() or "setup_app.py failed")

    progress("install_app")
    app_src = project_dir / "dist" / f"{APP_NAME}.app"
    app_dst = Path("/Applications") / f"{APP_NAME}.app"
    if not app_src.is_dir():
        raise RuntimeError("Built app bundle not found")
    _install_app_bundle(app_src, app_dst)

    return {
        "ok": True,
        "installed_path": str(app_dst),
        "project_dir": str(project_dir),
        "release_tag": release_tag,
    }


class UpgradeAlreadyInProgress(Exception):
    """Raised when a second upgrade is requested while one is active."""


class UpgradePreflightError(Exception):
    """Raised when preflight checks block an upgrade."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass
class UpgradeStatus:
    """Mutable upgrade progress snapshot."""

    running: bool = False
    step: str = "idle"
    message: str = ""
    ok: bool | None = None
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "running": self.running,
            "step": self.step,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "completed": self.step == "done" and self.ok is True,
        }


class UpgradeService:
    """Owns guarded background upgrade runs (git pull, uv sync, rebuild .app)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = UpgradeStatus()
        self._thread: threading.Thread | None = None

    def is_running(self) -> bool:
        with self._lock:
            return self._status.running

    def get_status(self) -> UpgradeStatus:
        with self._lock:
            return UpgradeStatus(
                running=self._status.running,
                step=self._status.step,
                message=self._status.message,
                ok=self._status.ok,
                error=self._status.error,
                result=dict(self._status.result),
            )

    def _claim(self) -> None:
        with self._lock:
            if self._status.running:
                raise UpgradeAlreadyInProgress()
            self._status = UpgradeStatus(
                running=True,
                step="preflight",
                message="Starting update…",
                ok=None,
            )

    def _abort_claim(self) -> None:
        with self._lock:
            self._status = UpgradeStatus()

    def _set_progress(self, step: str, message: str) -> None:
        with self._lock:
            self._status.step = step
            self._status.message = message

    def _finish(self, *, ok: bool, error: str | None = None, result: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._status.running = False
            self._status.ok = ok
            self._status.error = error
            if result:
                self._status.result = result
            if ok:
                self._status.step = "done"
                self._status.message = "Update installed — quit and reopen Escriba"

    def claim(self) -> None:
        """Claim the upgrade slot. Caller must hold ``AppState._lock`` if needed."""
        self._claim()

    def abort_claim(self) -> None:
        """Release a claimed slot after a failed preflight."""
        self._abort_claim()

    def start_worker(
        self,
        release_tag: str | None,
        assets: tuple[dict[str, str], ...],
    ) -> None:
        """Run preflight (outside ``AppState`` lock) and spawn the worker thread."""
        project_dir = _preflight_upgrade(release_tag)
        worker = threading.Thread(
            target=self._run_upgrade_worker,
            args=(project_dir, release_tag, assets),
            daemon=True,
            name="escriba-upgrade",
        )
        with self._lock:
            self._thread = worker
        worker.start()

    def try_begin(
        self,
        *,
        release_tag: str | None = None,
        assets: tuple[dict[str, str], ...] = (),
    ) -> None:
        """Claim and start an upgrade (CLI / tests without ``AppState`` lock).

        Raises:
            UpgradeAlreadyInProgress: When another upgrade is running.
            UpgradePreflightError: When preflight checks fail.
        """
        try:
            self.claim()
            self.start_worker(release_tag, assets)
        except Exception:
            self.abort_claim()
            raise

    def _run_upgrade_worker(
        self,
        project_dir: Path,
        release_tag: str | None,
        assets: tuple[dict[str, str], ...],
    ) -> None:
        try:
            result = _execute_upgrade(
                project_dir,
                release_tag=release_tag,
                assets=assets,
                on_progress=self._set_progress,
            )
        except Exception as exc:
            self._finish(ok=False, error=_safe_upgrade_error(exc))
        else:
            self._finish(ok=True, result=result)


def run_upgrade_blocking(
    *,
    recording_active: bool = False,
    release_tag: str | None = None,
    assets: tuple[dict[str, str], ...] = (),
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the upgrade synchronously (CLI)."""
    if recording_active:
        raise UpgradePreflightError("Stop recording before updating")
    project_dir = _preflight_upgrade(release_tag)
    try:
        return _execute_upgrade(
            project_dir,
            release_tag=release_tag,
            assets=assets,
            on_progress=on_progress,
        )
    except Exception as exc:
        raise RuntimeError(_safe_upgrade_error(exc)) from exc
