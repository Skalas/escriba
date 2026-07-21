# Sprint plan — unify AppState recording seam + split mic-activation poll

> Entry doc for `metate-prep`. Selected from discover: **merge 2 + 3**.
> Mode hint: **HOLD** on recording single-writer / stop-path parity; **REDUCE** on
> `_check_mic_activation` decomposition (behavior-preserving).

## Goal

Harden the recording start/stop path so menubar and HTTP share one clear
`AppState` claim model (especially stop: async menubar vs sync HTTP), then thin
`_check_mic_activation` into detect / decide / act helpers so the poll loop stays
testable and ready for a later calendar auto-start sprint (#193) without growing
another god-method.

## Why now

- **#2** — Graph: dual stop paths (`_Handler._stop_recording` sync vs menubar
  `_begin_stop_recording_session` async + `finish_stop_recording`). Calendar
  auto-start (#193) would add a third caller — harden the seam first.
- **#3** — `_check_mic_activation` is high fan-out (~21 outbound); same loop
  calendar scheduling would extend. REDUCE now while mic auto-record stays green.

## Scope note

Parents / links: structural prep for [#193](https://github.com/Skalas/escriba/issues/193);
does **not** implement calendar auto-start.

**In scope**

- Unify start/stop claiming around `AppState.try_start_recording` /
  `begin_stop_recording` / `finish_stop_recording` (or equivalent single-writer API)
- Align HTTP and menubar stop so both honor the same claim/finish contract
- Extract detect / decide / act (or similarly named) helpers from
  `_check_mic_activation` without changing mic auto-record behavior
- Tests for concurrent start, stop claim, and mic auto-record debounce paths
- Docs only if public behavior/docs drift (prefer minimal)

**Out of scope**

- Full calendar auto-start / Settings for calendar scheduling (#193)
- `do_GET` decomposition (discover #4)
- #105 P2 pile (MLX resample, Swift XCTest, schema_version, …)
- Daemon → AppState fold (separate HOLD; do not expand this sprint)

## Definition of Done

Done when: menubar and HTTP start/stop go through one `AppState` single-writer
contract (no divergent “sync stop skips claim” hole); mic auto-record
start/stop/debounce behavior is unchanged and covered by tests; `_check_mic_activation`
is a thin orchestrator over extracted helpers; ship gate green. #193 remains filed
and untouched functionally.

## Seed test matrix

### Strand A — AppState recording seam (#2) · HOLD

| ID | Criterion |
|----|-----------|
| T1 | Start: menubar + HTTP both use `try_start_recording` (or documented single entry); concurrent start still single-writer |
| T2 | Stop: both paths use `begin_stop_recording` + `finish_stop_recording` (or equivalent); no path stops the session while bypassing the claim |
| T3 | Already-stopping / no-session stop is idempotent and safe (clear errors, no double-finish crash) |
| T4 | Focused tests for concurrent start and stop-claim races (extend existing TG1/T1-style tests) |

### Strand B — Mic-activation poll REDUCE (#3) · REDUCE

| ID | Criterion |
|----|-----------|
| T5 | `_check_mic_activation` decomposed into detect / decide / act (names flexible); public menubar behavior unchanged |
| T6 | Auto-start on sustained mic-active and auto-stop on debounced inactive still pass (unit and/or existing call-detection tests) |
| T7 | Prompt vs auto `start_mode` still honored; cooldown / dismiss semantics unchanged |
| T8 | No intentional calendar-event start wired yet (assert #193 still out of this diff) |

### Strand C — Hygiene · HOLD

| ID | Criterion |
|----|-----------|
| T9 | ROADMAP / brief note: seam hardened; #193 still next product sprint when picked |

## Seed H-matrix (plan prose only — no `smoke.humanGates` in profile)

| ID | Type | What the human does |
|----|------|---------------------|
| H1 | live | With mic auto-record enabled: one real call-ish cycle starts and stops as before |
| H2 | live | Manual Record + dashboard Stop still clean (no stuck “stopping”, no double session) |
| H3 | other | Confirm calendar auto-start still off (no surprise starts from events) |

## Mode

Prep finalizes mode. Hint: **HOLD** overall (correctness of the recording mutex);
Strand B is **REDUCE** (structure only).

## Next ceremony

Hand off to **`metate-prep`**, which reads this file, files the T-matrix issue ledger,
and cuts the working branch from `main`.
