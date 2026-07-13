# Sprint plan — session-scoped notes + local-LLM timeouts + capture decomposition

> Entry doc for `metate-prep`. Selected from the discover slate: candidates **1 + 3 + 4**
> merged into one sprint.
> Mode hint: **HOLD** overall (with a REDUCE strand on the capture spine). Fix correctness
> and structure; do not expand product surface (calendar / soak stay out of scope).

## Goal

Make the live notes path session-correct, stop cold local-model loads from being misread as
generation timeouts, and shrink `run_streaming_capture` so future audio fixes land on testable
units — without changing capture behavior in this sprint.

## Why now

- **#125** (filed 2026-07-13): live notepad + notes-output are global DOM state; notes bleed
  across view switches, auto-record / menubar starts, and Enhance/generate, then autosave onto
  the wrong session. Blocks a trustworthy soak of the notes loop.
- **#108**: cached-but-cold `mlx_lm` load still shares the 300s generation deadline; legitimate
  slow loads kill the worker with no notes.
- **#103**: `run_streaming_capture` is the graph hotspot (cognitive ~639, ~997 LOC, fan-out 49);
  decomposition is the prerequisite for safe audio fixes and unit coverage.

## Scope note

Most work already has GitHub issues. `metate-prep` should link **#125**, **#108**, and **#103**
rather than filing duplicates, then create only missing ledger entries for test-matrix rows
without an existing issue.

Out of scope this sprint: real-meeting soak / clean-install (#2), calendar spike (#64),
model-download extraction (#109), P2 checklist (#105), sidebar clip (#87).

## Definition of Done

Done when: notepad and notes-output always reflect the session on display (every start route
and view transition); local inference separates load vs generation deadlines so a cold load
cannot be reported as a hung generate; `run_streaming_capture` is decomposed into named units
with behavior preserved and focused unit tests on the new seams. `uv run ruff check .`,
`uv run mypy .`, and `uv run pytest` green.

## Seed test matrix

### Strand A — Session-scoped live notes (#125) · HOLD

| ID | Criterion |
|----|-----------|
| T1 | On every displayed `session_id` change (poll/`syncStatus`, `showLiveView`, back-to-live), `#live-notepad` loads that session’s authoritative `user_notes` (empty for a brand-new session) and `#notes-output` / legend reset |
| T2 | Recordings started via auto-record, menubar, or poll-detected active session (not only the dashboard Start button) reconcile notepad state the same way |
| T3 | Enhance / generate never reads or writes notes belonging to a prior session; autosave refuses writes when the notepad’s keyed `session_id` ≠ current session |
| T4 | SPA regression: view-switch + different-route start + generate-then-switch do not cross-contaminate notes |

### Strand B — Local inference load vs generation timeout (#108) · HOLD

| ID | Criterion |
|----|-----------|
| T5 | Model load (`mlx_lm.load` / weight load into RAM) uses a separate deadline (generous or unbounded with progress), not the generation budget |
| T6 | Generation-only timeout still kills a hung generate and surfaces a clear timeout |
| T7 | Cached-but-cold large model + long transcript still produces notes (load no longer eats the generation window) |

### Strand C — Decompose `run_streaming_capture` (#103) · REDUCE

| ID | Criterion |
|----|-----------|
| T8 | Extract `CaptureSupervisor` (Swift-CLI spawn/monitor/restart + stderr drain) and `ChunkPump` (buffer accumulation, chunk slicing, rate/duration accounting); `run_streaming_capture` becomes thin orchestration |
| T9 | Behavior-preserving: existing live-capture / mixing / restart tests stay green; no intentional audio-behavior change |
| T10 | Unit tests cover the new chunking and restart seams (the extraction’s reason to exist) |

## Suggested issue mapping for prep

- Strand A → [#125](https://github.com/Skalas/escriba/issues/125) (split T1–T4 only if prep’s granularity needs one issue per matrix row)
- Strand B → [#108](https://github.com/Skalas/escriba/issues/108)
- Strand C → [#103](https://github.com/Skalas/escriba/issues/103)

## Explicit non-goals

- Calendar-driven recording / Up-next (#64)
- Human soak + clean-install verification
- Moving model-download orchestration out of the presentation layer (#109)
- Opportunistic P2 backlog (#105, #87)
