# Sprint plan — docs refresh + P2 leftovers + install single-source + release-CI hygiene

> Entry doc for `metate-prep`. Selected from discover: **2 + 3 + 4 + 5**.
> Mode hint: **REDUCE** (docs, P2 micro-bundle, CI hygiene) with a thin **HOLD** on
> install-path single-sourcing. Calendar spike (**#64**) explicitly deferred.

## Goal

Bring README/contributor docs in line with shipped features; clear another bounded
slice of ROADMAP P2 leftovers (ex-#105); either single-source the mutable install half
or keep intentional diffs with a sharper contract; and close the release-CI hygiene gap
left by historical red runs on `v1.3.0`/`v1.3.1`.

## Why now

- **#2** — README CLI/architecture lag `check-update`/`update` and the real module tree.
- **#3** — #105 is closed, but ROADMAP still lists deferred P2 (MLX resample, Swift signal
  handler, triple-flush, persistence indexes, …).
- **#4** — `install_paths.py` inventoried parity; optional to collapse drift before more cuts.
- **#5** — `--clobber` is on `main`; tag-era runs stay red; assets already fixed — document
  or one-shot retag hygiene so the next cut is obviously green.

## Scope note

No parent open issue required for docs/CI; file a fresh P2 checklist issue if triage wants
tracker visibility. **Out of scope:** calendar Up-next / auto-start (#64), Sparkle,
framework creep on the SPA.

## Definition of Done

Done when: README CLI + architecture match reality; a bounded P2 subset is fixed with
tests or rationale and ROADMAP deferred list updated; install path is either shared or
explicitly contracted; release-CI story is documented (and optionally proven green on the
next publish path). Ship gate green.

## Seed test matrix

### Strand A — README / docs (#2) · REDUCE

| ID | Criterion |
|----|-----------|
| T1 | README Usage/CLI lists real commands (`app`, `check-update`, `update`, `download-model`, `daemon`, `watch-calendar`, …) |
| T2 | Architecture / module tree matches current `src/escriba` (no stale `streaming_mps` unless still present) |
| T3 | Feature bullets mention in-app updates + auto-record (opt-in) without inventing unshipped claims |
| T4 | CONTRIBUTING or ROADMAP note for release asset upload / `--clobber` if not only in workflow |

### Strand B — P2 leftovers micro-bundle (#3) · REDUCE

| ID | Criterion |
|----|-----------|
| T5 | Triage ROADMAP deferred list; pick 2–4 cheap items (prefer: config triple-flush clarity, Swift signal-handler if small, persistence index, or narrow `except` — **not** full MLX resample unless trivial) |
| T6 | Each pulled item has a focused test or documented no-test rationale |
| T7 | ROADMAP / optional new issue updated: done vs still deferred |

### Strand C — Install single-source (#4) · HOLD

| ID | Criterion |
|----|-----------|
| T8 | Decision recorded: shared entry **or** keep intentional diffs |
| T9 | If shared: one path invoked by in-app upgrade and documented for `install.sh` / `make install`; if not: short contract in `install_paths.py` / ROADMAP with drift trigger |
| T10 | `make install` still produces `/Applications/Escriba.app`; upgrade dirty-tree refuse unchanged |

### Strand D — Release-CI hygiene (#5) · REDUCE

| ID | Criterion |
|----|-----------|
| T11 | Document that `v1.3.0`/`v1.3.1` Actions failures are stale (pre-`--clobber` tag workflow); assets were replaced via clobber upload |
| T12 | Optional: one-shot proof (workflow_dispatch green on `main`, or notes for next `gh release create` without pre-attaching assets so CI owns upload) |

## Suggested issue mapping for prep

- Strand A → docs chore issues
- Strand B → fresh P2 checklist issue (since #105 closed) or ROADMAP-only ledger rows
- Strand C → chore under install/upgrade
- Strand D → docs/chore (may ship with T4/T11 without a long-lived issue)

## Explicit non-goals

- Calendar-driven recording / Up-next (#64)
- Closing every ROADMAP P2 item in one sprint
- Rewriting install.sh into Python wholesale unless the shared-entry decision requires a thin wrapper
