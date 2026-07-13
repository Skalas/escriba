# Sprint plan — post-split auto-rename + soak gates + download extraction + v1.3.0 tag

> Entry doc for `metate-prep`. Selected from the discover slate: candidates **1 + 2 + 4 + 5**
> merged into one sprint.
> Mode hint: **HOLD** overall (REDUCE for model-download extraction; release chore for the tag).
> Do not expand calendar (#64) this cycle.

## Goal

Make post-split session titles auto-rename reliably (the real #137 failure), close the
remaining human release gates (soak + clean install), extract model-download orchestration
out of the HTTP presentation layer (#109), and cut/tag `v1.3.0` so the shipped code matches
the docs.

## Clarification on #137 (candidate 1)

The reporter does **not** mind `(part 1)` / `(part 2)` as interim labels. The failure is that
halves **did not automatically rename themselves** after split (post-split
`_regenerate_title` / auto-name path). Nested `(part N)` stacking may still be cleaned as a
small defensive side fix if cheap, but it is **not** the DoD of this strand.

## Why now

- **#137** (filed 2026-07-13): live sidebar left with stacked/default part names after split;
  user expectation is LLM (or other) auto-rename of both halves.
- **Aftercare / roadmap**: real-meeting soak + clean-install are still the open 1.0.x human
  gates after the 1.3.0 code merge.
- **#109**: download lifecycle still lives in `server.py` handlers — structural debt that
  blocks CLI/daemon reuse and testability.
- **v1.3.0 tag**: code for notes/timeouts/capture is on `main` but version metadata still
  reads `1.2.0`; Unreleased changelog entries need a dated release.

## Scope note

Link existing issues where they match: **#137** (reframed), **#109**. Soak/clean-install and
the release cut may need new ledger rows from prep. Out of scope: calendar spike (#64),
sidebar clip (#87), P2 checklist (#105), CaptureSupervisor further split, server
`append-notes`.

## Definition of Done

Done when: after a split, both halves receive auto-generated titles when auto-name is enabled
(and failures are visible/logged, not silent forever-stuck part labels); soak + clean-install
are executed and documented; model download is owned by an application-layer service shared
by HTTP + CLI; `v1.3.0` is bumped, changelog-dated, tagged, and pushed. Ship gate green.

## Seed test matrix

### Strand A — Post-split auto-rename (#137) · HOLD

| ID | Criterion |
|----|-----------|
| T1 | With auto-name enabled, splitting a session regenerates titles for **both** halves (not left on `(part N)` forever when generation succeeds) |
| T2 | Auto-rename failure is observable (log + optional UI signal) without blocking the split response forever; fallback part names remain valid |
| T3 | Manual rename still persists and refreshes the sidebar immediately |
| T4 | Tests cover success path + failure/degraded path for post-split title regeneration |

Optional (non-blocking): strip/avoid nested `(part N)` when re-splitting an already-named half.

### Strand B — Human release gates · HOLD

| ID | Criterion |
|----|-----------|
| T5 | Real-meeting soak: record → transcribe → summarize without manual rescue; results documented |
| T6 | Clean install-from-scratch (one-liner → `/Applications`) verified; results documented |
| T7 | Any soak/install blockers filed as issues (or confirmed none) |

### Strand C — Model-download extraction (#109) · REDUCE

| ID | Criterion |
|----|-----------|
| T8 | `ModelDownloadService` (or equivalent) owns claim/cancel/progress/completion; HTTP handler is thin |
| T9 | CLI `download-model` (and any daemon path) reuses the same service |
| T10 | Unit tests cover the service without spinning the full HTTP server |

### Strand D — Cut `v1.3.0` · HOLD (release)

| ID | Criterion |
|----|-----------|
| T11 | Version unified at `1.3.0` across `pyproject.toml`, `src/escriba/__init__.py`, `uv.lock` |
| T12 | CHANGELOG `[1.3.0]` dated; Unreleased cleared appropriately; ROADMAP marks 1.3.0 shipped |
| T13 | Tag `v1.3.0` created and pushed (prep/ship timing: tag at ship after gate green) |

## Suggested issue mapping for prep

- Strand A → [#137](https://github.com/Skalas/escriba/issues/137) (update body/title if needed to match auto-rename focus)
- Strand B → new issues or a single soak/install checklist issue (prep decides granularity)
- Strand C → [#109](https://github.com/Skalas/escriba/issues/109)
- Strand D → new release chore issue(s) or fold into ship without separate GitHub issues if prep prefers

## Explicit non-goals

- Calendar-driven recording / Up-next (#64)
- Making `(part N)` stacking the primary bug (reporter OK with part labels as interim)
- Sidebar title clipping (#87)
- P2 review checklist (#105)
