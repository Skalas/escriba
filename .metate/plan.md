# Sprint plan — in-app updates + docs sync + P2 cleanups + sidebar clip

> Entry doc for `metate-prep`. Selected from the discover slate: candidates **1 + 3 + 5 + 4**.
> Mode hint: **EXPAND** for updates (#152); **HOLD** for docs sync and sidebar (#87);
> **REDUCE**/opportunistic for P2 checklist (#105). No calendar (#64) this cycle.

## Goal

Ship an in-app update check with a one-click install path so users on `/Applications`
learn about new releases; sync post-soak docs and push local `main`; clear a scoped
slice of the P2 review checklist (#105); and fix the sidebar session-title clip (#87).

## Why now

- **#152** — filed after `v1.3.0`; installers have no in-app upgrade path beyond re-running
  `install.sh`.
- **Docs/push** — human soak signed off but ROADMAP still says soak is open; `main` is
  ahead of origin by the metate gitignore commit.
- **#105** — opportunistic P2 bundle; pull only items adjacent to this sprint’s surface.
- **#87** — open UX bug on the session list; ships next to the update banner (same SPA).

## Scope note

Link **#152**, **#105**, **#87**. Docs/push may be ledger rows without new GitHub issues.
Out of scope: calendar spike (#64), Swift XCTest, CaptureSupervisor further split,
server `append-notes`.

## Definition of Done

Done when: a running older build can detect a newer GitHub release and offer a one-click
update that reuses the installer path; ROADMAP reflects soak complete and `main` is pushed;
selected #105 items are fixed with tests where core-loop adjacent; session titles no longer
clip mid-line under sticky date headers. Ship gate green.

## Seed test matrix

### Strand A — In-app updates (#152) · EXPAND

| ID | Criterion |
|----|-----------|
| T1 | `GET` update-check (or extended `/api/version`) compares `__version__` to GitHub `releases/latest`; returns current/latest/update_available/urls |
| T2 | Dashboard (About and/or banner) notifies when update available; dismiss/snooze until next version |
| T3 | One-click Update runs guarded upgrade (pull/ff, `uv sync`, rebuild `.app` / refresh capture asset as needed); progress + success/fail visible; prompts relaunch |
| T4 | Offline / GitHub down → fail-soft (no error spam); mutating install respects CSRF |
| T5 | CLI `escriba check-update` and/or `escriba update` for headless parity |

### Strand B — Post-soak docs + push · HOLD

| ID | Criterion |
|----|-----------|
| T6 | ROADMAP (and any stale “Where we are”) state soak + clean-install as done; next pointer → updates and/or calendar |
| T7 | Push local `main` commits (incl. metate gitignore) to origin |

### Strand C — P2 checklist slice (#105) · REDUCE

| ID | Criterion |
|----|-----------|
| T8 | Triage #105; implement a bounded subset that is cheap adjacent to this sprint (document which items deferred) |
| T9 | Each pulled item has a focused test or documented no-test rationale |

### Strand D — Sidebar title clip (#87) · HOLD

| ID | Criterion |
|----|-----------|
| T10 | Session titles in the sidebar do not clip mid-line above/under sticky date headers |
| T11 | Visual check light + dark (SPA or manual); no regression to rename/select affordances |

## Suggested issue mapping for prep

- Strand A → [#152](https://github.com/Skalas/escriba/issues/152) (split T1–T5 if granularity requires)
- Strand B → optional chore issues or ship-only
- Strand C → [#105](https://github.com/Skalas/escriba/issues/105)
- Strand D → [#87](https://github.com/Skalas/escriba/issues/87)

## Explicit non-goals

- Calendar-driven recording / Up-next (#64)
- Forced/silent auto-update without confirmation
- Sparkle / notarized delta packages (reuse installer path first)

## Appendix — #105 deferred (2026-07-13)

Implemented this sprint: Range-416, config `mkstemp`, raw `system_prompt` / no default persist, `unique_export_path` TOCTOU via exclusive create, meeting-app `process_names` (pre-existing #112). Still open from #105: persistence indexes + `schema_version` runner, denormalized `segment_count`, batched writes, config hot-reload, structlog migration, typed handler responses, streaming summaries, server `append-notes`, CaptureSupervisor stderr split, Swift XCTest.
