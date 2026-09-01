"""#215: the MLX inference worker must never outlive the app that spawned it.

Covers the three layers of the fix: the worker-side parent-death watchdog (the
only one that survives SIGKILL), the explicit pool shutdown, and the startup
sweep that reaps workers stranded by builds predating this change.
"""

from __future__ import annotations

import signal
import subprocess
import sys

import pytest

from escriba.summarize import llm_summary


# ---------------------------------------------------------------------------
# Worker-side parent-death watchdog
# ---------------------------------------------------------------------------


class _Exited(Exception):
    """Stands in for os._exit, which a test cannot let run."""


@pytest.fixture
def stub_exit(monkeypatch):
    def _fake_exit(code):
        raise _Exited(code)

    monkeypatch.setattr(llm_summary.os, "_exit", _fake_exit)


def test_watchdog_exits_when_reparented(monkeypatch, stub_exit):
    """A worker adopted by launchd self-reaps instead of holding 14 GB forever."""
    monkeypatch.setattr(llm_summary.os, "getppid", lambda: 1)
    monkeypatch.setattr(llm_summary.time, "sleep", lambda _s: None)

    with pytest.raises(_Exited) as excinfo:
        llm_summary._exit_when_orphaned(initial_ppid=4242, poll_seconds=0.0)

    assert excinfo.value.args[0] == 0


def test_watchdog_survives_a_launchd_owned_parent(monkeypatch, stub_exit):
    """ppid == 1 alone must not trigger: the app itself is owned by launchd.

    The watchdog compares against the parent seen at start, so a worker whose
    parent was already PID 1 keeps running rather than killing live inference.
    """
    monkeypatch.setattr(llm_summary.os, "getppid", lambda: 1)

    polls = 0

    def _fake_sleep(_seconds):
        nonlocal polls
        polls += 1
        if polls >= 3:
            raise KeyboardInterrupt  # break the loop without exiting

    monkeypatch.setattr(llm_summary.time, "sleep", _fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        llm_summary._exit_when_orphaned(initial_ppid=1, poll_seconds=0.0)
    assert polls == 3


def test_watchdog_thread_starts_once_per_worker(monkeypatch):
    started = []

    class _FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs)

    monkeypatch.setattr(llm_summary, "_watchdog_started", False)
    monkeypatch.setattr(llm_summary.threading, "Thread", _FakeThread)

    llm_summary._ensure_orphan_watchdog(4242)
    llm_summary._ensure_orphan_watchdog(4242)
    llm_summary._ensure_orphan_watchdog(4242)

    assert len(started) == 1
    assert started[0]["daemon"] is True
    assert started[0]["args"] == (4242, llm_summary._ORPHAN_WATCHDOG_POLL_SECONDS)


def test_watchdog_baseline_comes_from_the_parent_not_the_child(monkeypatch):
    """The parent PID must be passed in, never read via getppid() in the child.

    If the parent dies during spawn, the child's own getppid() already reads 1,
    so a child-derived baseline would equal 1 forever and the watchdog would
    never fire — disarming it for precisely the worker that got orphaned.
    """
    created = {}

    class _FakePool:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(llm_summary.concurrent.futures, "ProcessPoolExecutor", _FakePool)

    llm_summary._LocalInferenceProcess()._get_executor()

    assert created["initializer"] is llm_summary._worker_init
    assert created["initargs"] == (llm_summary.os.getpid(),)


def test_worker_init_arms_the_watchdog_with_the_given_parent(monkeypatch):
    armed = []
    monkeypatch.setattr(llm_summary, "_ensure_orphan_watchdog", armed.append)

    llm_summary._worker_init(4242)

    assert armed == [4242]


# ---------------------------------------------------------------------------
# Explicit pool shutdown
# ---------------------------------------------------------------------------


def test_shutdown_kills_the_running_worker():
    """shutdown() must reach _terminate_workers, not just cancel pending futures."""
    proc = llm_summary._LocalInferenceProcess()
    killed = []

    class _FakeExecutor:
        def shutdown(self, **_kwargs):
            pass

    monkeyed = _FakeExecutor()
    proc._executor = monkeyed
    proc._terminate_workers = lambda executor: killed.append(executor)

    proc.shutdown()

    assert killed == [monkeyed]
    assert proc._executor is None


def test_shutdown_is_safe_with_no_pool():
    llm_summary._LocalInferenceProcess().shutdown()  # must not raise


# ---------------------------------------------------------------------------
# Startup sweep
# ---------------------------------------------------------------------------


def _ps_output(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _stub_ps(monkeypatch, stdout: str) -> None:
    def _fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)

    monkeypatch.setattr(llm_summary.subprocess, "run", _fake_run)


WORKER = "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=18)"
TRACKER = "from multiprocessing.resource_tracker import main;main(13)"


def test_sweep_finds_launchd_adopted_workers(monkeypatch):
    _stub_ps(
        monkeypatch,
        _ps_output(
            f"  501     1 {sys.prefix}/bin/python3 -c {WORKER}",
            f"  502     1 {sys.prefix}/bin/python3 -c {TRACKER}",
        ),
    )
    assert llm_summary._find_orphaned_worker_pids() == [501, 502]


def test_sweep_spares_a_live_apps_worker(monkeypatch):
    """A worker with a living parent belongs to a running Escriba — leave it."""
    _stub_ps(
        monkeypatch,
        _ps_output(f"  501  9000 {sys.prefix}/bin/python3 -c {WORKER}"),
    )
    assert llm_summary._find_orphaned_worker_pids() == []


def test_sweep_spares_other_venvs(monkeypatch):
    _stub_ps(
        monkeypatch,
        _ps_output(f"  501     1 /opt/other/.venv/bin/python3 -c {WORKER}"),
    )
    assert llm_summary._find_orphaned_worker_pids() == []


def test_sweep_spares_non_worker_processes(monkeypatch):
    """The app itself runs from the venv under launchd; it is not a pool worker."""
    _stub_ps(
        monkeypatch,
        _ps_output(f"  501     1 {sys.prefix}/bin/python3 -m escriba app"),
    )
    assert llm_summary._find_orphaned_worker_pids() == []


def test_sweep_never_targets_itself(monkeypatch):
    pid = llm_summary.os.getpid()
    _stub_ps(
        monkeypatch,
        _ps_output(f"  {pid}     1 {sys.prefix}/bin/python3 -c {WORKER}"),
    )
    assert llm_summary._find_orphaned_worker_pids() == []


def test_sweep_survives_ps_failure(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise OSError("ps unavailable")

    monkeypatch.setattr(llm_summary.subprocess, "run", _boom)
    assert llm_summary._find_orphaned_worker_pids() == []
    assert llm_summary.reap_orphaned_inference_workers() == 0


def test_reap_terminates_then_kills_survivors(monkeypatch):
    monkeypatch.setattr(llm_summary, "_find_orphaned_worker_pids", lambda: [501, 502])
    monkeypatch.setattr(llm_summary.time, "sleep", lambda _s: None)

    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(
        llm_summary, "_signal_pid", lambda pid, sig: signalled.append((pid, sig))
    )
    # 501 dies on SIGTERM; 502 ignores it.
    monkeypatch.setattr(llm_summary, "_pid_alive", lambda pid: pid == 502)

    assert llm_summary.reap_orphaned_inference_workers(grace_seconds=0.0) == 2

    assert (501, signal.SIGTERM) in signalled
    assert (502, signal.SIGTERM) in signalled
    assert (502, signal.SIGKILL) in signalled
    assert (501, signal.SIGKILL) not in signalled


def test_reap_is_a_noop_when_clean(monkeypatch):
    monkeypatch.setattr(llm_summary, "_find_orphaned_worker_pids", lambda: [])

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("must not signal anything")

    monkeypatch.setattr(llm_summary, "_signal_pid", _unexpected)
    assert llm_summary.reap_orphaned_inference_workers() == 0


def test_shutdown_does_not_block_on_an_in_flight_inference():
    """Quit must not wait on the lock `run` holds for the whole generation.

    `run` keeps `_lock` across a future wait of up to ~16 minutes. If shutdown
    blocked on it, quitting mid-summary would hang the app — and a user who then
    force-quits strands exactly the worker this is meant to reap.
    """
    import threading
    import time

    proc = llm_summary._LocalInferenceProcess()
    killed = []

    class _FakeExecutor:
        def shutdown(self, **_kwargs):
            pass

    proc._executor = _FakeExecutor()
    proc._terminate_workers = lambda executor: killed.append(executor)

    holder_has_lock = threading.Event()
    release = threading.Event()

    def _hold_lock():
        with proc._lock:
            holder_has_lock.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert holder_has_lock.wait(timeout=5)

    started = time.monotonic()
    proc.shutdown(lock_timeout=0.2)
    elapsed = time.monotonic() - started

    release.set()
    holder.join(timeout=5)

    assert elapsed < 2.0, f"shutdown blocked for {elapsed:.1f}s"
    # The worker is killed regardless; the in-flight run resets the broken pool.
    assert len(killed) == 1


def test_shutdown_prevents_a_new_worker_from_being_spawned(monkeypatch):
    """The worst #215 regression: quitting mid-summary spawning a *fresh* 14 GB worker.

    shutdown() that loses the lock race kills the worker and leaves the pool
    set; the blocked run() then sees BrokenProcessPool (a RuntimeError), and
    without the shutting-down flag it would loop, build a new pool and spawn a
    new worker — after quit_app has already returned — orphaning it for good.
    """
    proc = llm_summary._LocalInferenceProcess()
    spawned = []

    class _FakePool:
        def __init__(self, **_kwargs):
            spawned.append(self)

        def submit(self, *_a, **_kw):
            raise AssertionError("must not submit during shutdown")

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(llm_summary.concurrent.futures, "ProcessPoolExecutor", _FakePool)

    proc.shutdown()
    assert proc.run("p", "m", 10, False) is None
    assert spawned == [], "a new inference worker was spawned during shutdown"


def test_get_executor_refuses_after_shutdown():
    proc = llm_summary._LocalInferenceProcess()
    proc.shutdown()
    with pytest.raises(RuntimeError, match="shutting down"):
        proc._get_executor()


def test_sweep_skipped_outside_a_venv(monkeypatch):
    """Without a venv, sys.prefix is shared and cannot prove ownership."""
    monkeypatch.setattr(llm_summary.sys, "prefix", "/usr")
    monkeypatch.setattr(llm_summary.sys, "base_prefix", "/usr")

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("must not enumerate processes outside a venv")

    monkeypatch.setattr(llm_summary.subprocess, "run", _unexpected)
    assert llm_summary._find_orphaned_worker_pids() == []


def test_terminate_workers_survives_a_concurrent_worker_map_mutation(caplog):
    """shutdown() snapshots _processes without the lock, racing the pool manager."""

    class _ExplodingMap(dict):
        def values(self):
            raise RuntimeError("dictionary changed size during iteration")

    class _FakeExecutor:
        _processes = _ExplodingMap({1: object()})

    # Must not raise; a failed snapshot is logged, not swallowed silently.
    llm_summary._LocalInferenceProcess._terminate_workers(_FakeExecutor())
    assert "Could not snapshot inference workers" in caplog.text
