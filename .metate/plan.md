# Sprint plan — verify in-app updates + install parity + P2 + release docs

> Entry doc for `metate-prep`. Selected from discover: **1 ← (3+4+5)** —
> end-to-end update verification as the spine; fold in single-source install (#3),
> another #105 P2 slice (#4), and CHANGELOG/release housekeeping (#5).
> Mode hint: **HOLD** (prove what we shipped); **REDUCE** on #105; light **HOLD** on docs.

## Goal

Prove that an older installed build detects a newer GitHub release **tag** and can offer
upgrade; reduce drift between `updates.py` and `install.sh` / Makefile; clear another
bounded #105 slice; fold Unreleased in-app-update notes into a dated release section
(and cut/tag if needed).

## Why now

- Discover #1 — human intent: verify updates via version-tag change after `v1.3.0` publish.
- Discover #3 — aftercare deferred install-path single-sourcing; trigger rises once live
  updates are exercised.
- Discover #4 — #105 still open after the last opportunistic slice.
- Discover #5 — in-app updates still sit under `[Unreleased]` while `v1.3.0` is Latest.

## Scope note

Parents / links: in-app updates follow-up (no open #152 — closed); [#105](https://github.com/Skalas/escriba/issues/105);
release/docs chore. **Out of scope:** calendar spike (#64), Sparkle, forced silent auto-update.

## Definition of Done

Done when: a build reporting an older `__version__` sees a newer `releases/latest` tag
(`update_available: true`) via CLI and Settings → Check for updates; install/upgrade step
parity is either shared or explicitly documented; selected #105 items are fixed with tests
or no-test rationale; CHANGELOG/ROADMAP no longer leave the updater under Unreleased while
Latest is already cut. Ship gate green.

## Seed test matrix

### Strand A — Update verification (discover #1) · HOLD

| ID | Criterion |
|----|-----------|
| T1 | Publish (or use) a release tag newer than the soak app’s `__version__` (e.g. `v1.3.1` or a deliberate prerelease marked latest for the test window) |
| T2 | App/CLI under older version reports `latest` = that tag and `update_available: true` |
| T3 | Settings → **Check for updates** surfaces banner/About “Update available”; dismiss/snooze still works |
| T4 | Guarded install path either completes to the new tag on a clean install tree **or** is proven via dry-run/status steps with documented human skip if dirty-tree refuse is hit |
| T5 | After bumping local `__version__` to match latest (or reinstall), check reports up-to-date |

### Strand B — Install path parity (discover #3) · HOLD

| ID | Criterion |
|----|-----------|
| T6 | Inventory `updates._execute_upgrade` vs `install.sh` / `make install` steps; gaps listed |
| T7 | Either one shared entry point for the mutable half **or** a short ADR/ROADMAP note of intentional diffs (no silent drift) |
| T8 | Regression: `make install` still produces `/Applications/Escriba.app`; update preflight still refuses dirty trees |

### Strand C — #105 P2 slice (discover #4) · REDUCE

| ID | Criterion |
|----|-----------|
| T9 | Triage open #105; implement a bounded subset (prefer items adjacent to config/server/export or cheap watch/LLM cleanups); document deferred remainder |
| T10 | Each pulled item has a focused test or documented no-test rationale |

### Strand D — Release docs (discover #5) · HOLD

| ID | Criterion |
|----|-----------|
| T11 | Move Unreleased in-app-update notes into a dated section (e.g. under `1.3.1` or amend release notes) matching the verification tag |
| T12 | ROADMAP “Where we are” / next pointer reflects verification done; calendar (#64) remains next product spike |

## Suggested issue mapping for prep

- Strand A → new child issues under a parent “Verify in-app updates” (or reopen-style follow-up)
- Strand B → child of that parent or standalone chore
- Strand C → [#105](https://github.com/Skalas/escriba/issues/105)
- Strand D → docs/chore issues or ship-only if tiny

## Explicit non-goals

- Calendar Up-next / auto-start (#64)
- Replacing GitHub Releases with Sparkle / notarized deltas
- Closing all of #105 in one sprint
