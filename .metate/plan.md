# Sprint plan — v1.0.1 release-blocker hardening

> Entry doc for `metate-prep`. Selected from the discover slate: candidates **1 + 2 + 3**
> merged into one sprint.
> Mode hint: **HOLD** — no new features; the P0/P1 backlog from the post-1.0.0 full-repo
> review governs the rigor. Fix the release-blocker surface; don't widen it.

## Goal

v1.0.0 shipped the code-side release hardening, then a full-repo review (issues #87–#105,
bundled in #105) surfaced a stack of P0/P1 correctness, reliability, and security bugs that
the release-readiness sprint did not cover. This sprint closes the **release-blocker tier**:
the core transcription pipeline must produce correct, in-sync output; recording must not leak
processes/handles or race itself; and the localhost attack surface must be closed.

Three strands, all "make 1.0 actually solid," all HOLD:

1. **Audio-capture correctness** — the primary pipeline currently produces garbage or
   desynced transcription under real conditions.
2. **Recording lifecycle** — subprocess/handle/GPU-worker leaks and start/restart races that
   accumulate into crashes over a session.
3. **Web-security pass** — CSRF + the P1 web-security cluster; the roadmap's "v1.0.0 security
   pass" trigger has fired.

**Enabler (in-strand, not separate scope):** `run_streaming_capture` (cognitive complexity
587, #103) is too tangled to patch safely — decompose it *as part of* Strand A so the audio
fixes land on a testable surface. Pull in only the decomposition the fixes require.

## Why now (signals)

- The post-1.0.0 review filed **6 P0** and **6 P1** bugs; none are covered by the v1.0.0
  hardening sprint, which was scoped to graceful-failure + path-disclosure only.
- Two P0s (#88 garbage transcription, #90 transcript desync) break the product's one job.
- The roadmap explicitly defers a **"v1.0.0 security pass"** trigger — #93/#94/#95/#96 fire it.
- The roadmap's 1.0.x gate wants a **real-meeting soak**; that soak is meaningless until the
  audio pipeline and recording lifecycle are correct.

## Scope note

The issues below **already exist on GitHub**. `metate-prep` should **link** each test-matrix
row to its existing issue (do not file duplicates); apply the `sprint` label and record them
in the issue ledger.

## Out of scope (deferred)

- **#97** `watch-calendar` no-op, **#99** GUI-path mlx fallback, **#64** calendar spike
  (discover candidate 4 — broken/incomplete features, not release-blockers).
- **#105** minor-cleanups bundle, **#87** sidebar title clip, concurrent note-gen race
  (discover candidate 5 — ride-along P2, pull in only if adjacent work makes it cheap).
- P2 backlog (persistence indexes, schema versioning, typing) — unchanged from roadmap.

---

## Definition of Done — test matrix

Done when: the core record → transcribe loop produces correct, in-sync output on real audio;
recording start/stop/failure leaves no orphaned processes, handles, or `active` DB rows; and
no state-changing endpoint accepts a cross-origin request. All proven by tests; the full ship
gate (ruff + mypy + pytest) green.

### Strand A — audio-capture correctness  *(enabler: decompose #103 as needed)*

- **T1** (#88) — PCM samples are normalized by the divisor matching their **actual bit depth**
  (32-bit input no longer divided by the 16-bit divisor). Test: a 32-bit PCM fixture
  transcribes to correct amplitude, not garbage.
- **T2** (#90) — a failed/dropped audio chunk **does not silently advance the transcript
  clock**; the timeline stays aligned (backfill silence or hold the clock). Test: a dropped
  chunk mid-stream leaves subsequent segment timestamps correct.
- **T3** (#92) — NaN/Inf samples are sanitized **before** the `Int16(...)` conversion in the
  Swift capture helper (no trap/crash). Test: a NaN/Inf-laden buffer is handled without a crash.
- **T4** (#104) — in `both` mode, mic and system streams are **rate-matched** (resampled to a
  common rate) before mixing; no progressive drift. Test: mismatched-rate inputs stay aligned
  over a multi-minute mix.

### Strand B — recording lifecycle (leaks & races)

- **T5** (#89) — `TranscriptionSession.start()` failure **releases** the subprocess and WAV
  handle and marks the DB row `failed` (never leaves it `active`). Test: a forced start
  failure leaves no live child, no open handle, and a non-`active` row.
- **T6** (#91) — the exhausted-retry path in `monitor_swift_cli` **kills** the Swift child
  (no orphan). Test: retries exhausted → child process is reaped.
- **T7** (#102) — `ScreenCaptureAudioCapture` start/restart is **single-spawn** and never
  skips cleanup (no double-spawn race). Test: rapid restart spawns exactly one capture.
- **T8** (#100) — a timed-out local inference **reaps** its GPU-loaded worker process (no
  leak). Test: an inference timeout terminates the worker.
- **T9** (#98) — the daemon **validates socket liveness** (doesn't trust a stale socket file);
  the CLI does not throw raw tracebacks after a crash. Test: a stale socket is detected and
  handled cleanly.

### Strand C — web-security pass (localhost attack surface)

- **T10** (#93, **P0**) — state-changing endpoints (`POST`/`PUT`/`DELETE`) **reject
  cross-origin requests** via an Origin/Host check (CSRF guard). Test: a request with a
  foreign Origin is refused; a same-origin request succeeds.
- **T11** (#95) — `escAttr` actually escapes for **attribute/inline-handler** context (the
  stored-XSS via `onclick` is closed). Test: a session title containing an `onclick`-breaking
  payload renders inert.
- **T12** (#94) — `PUT /api/config` **rejects newline injection** into `.env` values. Test: a
  value containing `\n KEY=evil` is rejected, not written as a second env line.
- **T13** (#96) — the Telegram bot token is **redacted** from logs on send failure. Test: a
  forced send failure logs no token substring.
- **T14** (#101) — watched-folder filenames **cannot inject argv** into the whisper
  `_build_command` (argv list, not shell; validated names). Test: a filename crafted as a flag
  is passed as a literal path, not an option.
