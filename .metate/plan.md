# Sprint plan - merged reliability, correctness, and automation hardening

> Entry doc for `metate-prep`. Selected from the discover slate: candidates **1 + 2 + 3 + 4 + 5**
> merged into one sprint.
> Mode hint: **HOLD** overall. This is a broad hardening sprint with one EXPAND-leaning
> automation strand; keep each fix issue-sized and prove it with tests.

## Goal

Close the current post-1.0.1 defect slate across the record -> transcribe -> summarize loop,
dashboard notes workflow, daemon IPC, and incomplete automation surfaces.

The sprint is intentionally broad because the open issues are clustered around the same product
promise: a real meeting should record, transcribe, save notes, stop cleanly, and optionally run
through automation without corrupting data, leaking processes, or silently dropping output.

## Why now

- Fresh P0/P1 GitHub issues from the 2026-07-08 full-repo review identify silent audio
  corruption, data-loss races, and lifecycle leaks.
- The roadmap still names real-meeting soak and clean-install verification as the remaining
  1.0.x manual gates; those gates are not meaningful while stop/finalization and live capture
  can corrupt data.
- The codebase graph shows the highest-risk surfaces are central: `run_streaming_capture`,
  `TranscriptionSession.stop`, daemon start/stop, and the dashboard session-detail workflow.
- The roadmap also calls out the next broken/incomplete-features sprint (#97/#99/#64); fold
  that into this cycle after the P0/P1 correctness work is bounded.

## Scope note

Most work already has GitHub issues. `metate-prep` should link existing issues rather than
filing duplicates where a matching issue exists, then create only missing ledger entries for
test rows without an existing issue.

## Definition of Done

Done when: stopping a recording is exception-safe and timeout-safe; live audio buffers and
backend resampling cannot silently corrupt transcript/audio; dashboard async note flows cannot
cross-write or lose edits; daemon IPC is single-writer and locally hardened; the incomplete
automation paths either work end-to-end or fail honestly. `uv run ruff check .`, `uv run mypy .`,
and `uv run pytest` are green.

## Test matrix

### Strand A - recording stop/finalization data safety

- **T1** (#114) - `TranscriptionSession.stop()` runs each cleanup step independently. A capture
  teardown exception cannot skip buffer flush, WAV close, export, or `db.stop_session`.
- **T2** (#115) - timed joins do not proceed as though the worker finished. If the process
  thread is still alive, the main thread does not concurrently flush/close the transcriber or
  WAV writer.
- **T3** (#115) - title refinement never starts a second local generation while an earlier title
  generation thread is still alive.
- **T4** (#115) - app quit does not close the DB or terminate the app while a recording stop is
  still completing.
- **T5** - tests force teardown failure, slow process-thread join, slow title generation, and
  quit-during-stop; all leave the session completed or clearly failed, never half-written.

### Strand B - live audio correctness and capture lifecycle

- **T6** (#110) - shared live audio buffers are protected by a clear locking discipline or
  replaced with a safe queue/ring buffer. Extend, slice/consume, and clear cannot race.
- **T7** (#111) - Swift CLI monitor backoff is interruptible via `stop_event`; a stop during
  backoff cannot restart and orphan a new capture process.
- **T8** (#116) - faster-whisper resamples non-16 kHz audio to 16 kHz before ndarray inference,
  matching the MLX backend behavior.
- **T9** (#113) - invalid WAV headers cannot produce zero-byte chunks or a CPU busy-loop.
- **T10** (#123) - faster-whisper early-return guards close chunk metrics, and MLX rejects
  unsupported sample widths instead of decoding them as 8-bit.
- **T11** - tests cover both-mode buffer concurrency, stop-during-restart, non-16 kHz WAV input,
  malformed WAV headers, and unsupported sample widths.

### Strand C - dashboard note and session async safety

- **T12** (#120) - stale `selectSession` responses cannot render into a newer selected session
  or cause notes from one session to be saved into another.
- **T13** (#121) - `generateSessionNotes()` appends to the current post-await notes content, or
  otherwise prevents editing while generation is in flight; saved edits are not lost.
- **T14** (#123) - `retranscribeSession` and other long session actions capture the session id
  before `await` and never refresh or mutate the wrong selected session on completion.
- **T15** (#123) - pending search/deep-link highlight state is cleared on `selectSession` early
  returns; stale highlight state cannot leak into a later unrelated session.
- **T16** - Playwright tests prove rapid A -> B selection, edit-during-generation, and
  navigate-during-retranscribe do not corrupt notes or UI state.

### Strand D - daemon IPC hardening

- **T17** (#117) - daemon start/stop check-and-set is guarded by a recording lock; concurrent
  `start-recording` commands can create at most one capture thread.
- **T18** (#117) - daemon command reads are framed or read to EOF; long valid JSON commands are
  not truncated by a single `recv(4096)`.
- **T19** (#117) - daemon socket directory and socket file are owner-only (`0700`/`0600` or
  equivalent), with tests that assert the final modes.
- **T20** - daemon tests cover concurrent starts, long commands, stop during active recording,
  and stale/permission-sensitive socket setup.

### Strand E - incomplete automation and notification reliability

- **T21** (#99) - GUI recording path has the same mlx -> faster-whisper fallback behavior as the
  hardened backend path; a missing MLX backend does not produce a no-transcriber session.
- **T22** (#97/#64) - calendar-driven recording is either made functional for the current
  supported path or explicitly disabled/marked unavailable in CLI/UI/docs instead of being a
  silent no-op.
- **T23** (#118) - watch-folder handles atomically written files via `on_moved` and allows a
  corrected replacement file after a failed attempt.
- **T24** (#119) - Telegram notifications cannot be dropped by Telegram Markdown parse errors
  from untrusted LLM text. Prefer plain text unless formatting is proven safe.
- **T25** (#122) - menubar auto-stop tracking id is cleared only after stop initiation succeeds;
  a failed stop attempt does not orphan an auto-started recording from future auto-stop checks.

### Strand F - bundled low-risk correctness polish

- **T26** (#123) - API edge cases return clear 4xx responses where appropriate: invalid Range
  start, chunked request body handling, unknown folder id, nonexistent merge ids, and TOML null
  config leaves.
- **T27** (#123) - SRT export cue numbers are contiguous after empty segments are skipped.
- **T28** (#123) - local summary timeout errors are handled and logged with the same clarity as
  Gemini/Claude timeout paths.
- **T29** (#123) - live transcript polling renders the empty state when segments reset instead
  of leaving stale transcript content on screen.

## Out of scope

- New product features beyond making the existing automation surfaces honest and reliable.
- Framework changes to the stdlib server or single-file SPA.
- Persistence-index/schema-version P2 backlog unless directly required by a selected test row.
- Real-meeting soak and clean-install verification themselves; those remain human-run gates
  after this plan lands and tests pass.
