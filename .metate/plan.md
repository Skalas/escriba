# Sprint plan — v1.0.0 release hardening

> Entry doc for `metate-prep`. Selected from the discover slate (candidates 1 + 2 + 4).
> Mode hint: **HOLD** — no new features; harden existing surface and make the release ready.

## Goal

Get Escriba release-ready for `v1.0.0` without widening the surface. Three strands, all
"readiness, not reach":

1. **Frontend error-paths + `/api/version` privacy** — close the v0.12.0 deferred items
   whose triggers fire at v1.0.0 (the SPA's `apiCall`/start-flow throw path, `saveNotes()`
   attribution, the saved-session generate auto-persist gap, and the absolute-path leak).
2. **CI-safe pytest** — make the swift capture integration test skip gracefully so
   `uv run pytest` is reliable headless / in CI.
3. **Docs + version + `uv.lock` audit** — release bookkeeping so metadata and onboarding
   match behavior.

Explicitly **out of scope** (deferred): P2 persistence schema-versioning + migration runner
(new infrastructure, not release-demanded); the calendar spike (#64); real-meeting soak and
clean-install verification remain human/manual smoke steps, not code sprint items.

## Why now (signals)

- Roadmap names `v1.0.0 — Release hardening` as **next up**; the milestone exists, empty.
- v0.12.0 deferred four items **with triggers**, three of which name "v1.0.0 hardening" /
  "v1.0.0 security pass" — those triggers fire with this sprint.
- The swift test hang is a named v1.0.0 DoD line blocking reliable CI.

## Definition of Done — test matrix

### Strand 1 — frontend error-paths + version privacy

- **T1** — `apiCall` (index.html) surfaces HTTP 4xx/5xx as a structured error result instead
  of returning `null`; every `if (!res.ok)` caller still behaves (no thrown exception on a
  failed call).
- **T2** — `/api/recording/start` failure shows the error and never throws or leaves a
  half-started recording; the start flow is safe when the server returns an error body.
- **T3** — `saveNotes()` (the notes + user-notes `Promise.all`) reports *which* POST failed
  rather than a single opaque failure; a partial failure is attributable in the UI.
- **T4** — `/api/version` no longer returns the absolute `project_dir` path (server.py:~820);
  build/version info stays, the filesystem path is removed or reduced to a non-sensitive form.
- **T5** — saved-session generate path auto-persists `notes_text` (the live path already
  does); regenerated notes on a saved session survive reload. *(4th deferred item; fold in.)*

### Strand 2 — CI-safe pytest

- **T6** — the swift `test_integration.py::test_capture_short` (and any sibling that needs a
  live input device) skips gracefully when no audio input device is available; `uv run pytest`
  runs green headless with no hang.

### Strand 3 — docs / version / lock audit

- **T7** — version strings are consistent across `pyproject.toml`, README, the app "about"/
  `/api/version`, and any hardcoded strings; bump to `1.0.0`.
- **T8** — clean install-from-scratch path is documented and the one-liner installer →
  `/Applications` flow is verified (or the gap is recorded).
- **T9** — `uv.lock` audited (no stale/unexpected entries); README/CLAUDE.md feature list and
  version current.

## Notes for prep

- Profile gates: fast `ruff + pytest`; ship `ruff + mypy + pytest`. Core-loop changes
  (`server.py`) ship with tests per the profile invariants.
- Strand 1 is mostly `index.html` + one `server.py` field; `apiCall` is the single fetch
  chokepoint every `res.ok` site rides on — change it once, verify callers.
- Strand 2 is a low-risk skip guard. Strand 3 is docs/metadata only.
