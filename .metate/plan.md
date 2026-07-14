# Sprint plan — atomic append-notes + P2 micro-bundle + knowledge adapters + thin calendar spike

> Entry doc for `metate-prep`. Selected from discover: **3 + 4 + 5 + thin 1**.
> Mode hint: **HOLD** on append-notes; **REDUCE** on P2; **EXPAND** on knowledge
> adapters; **EXPAND (spike-thin)** on calendar #64 — investigation + smallest home
> surface, not full auto-start.

## Goal

Land a server-side atomic `append-notes` path so concurrent Enhance cannot lose notes;
clear another cheap P2 slice from ROADMAP “Still open”; ship `webhook` + `custom-script`
knowledge adapters behind local-first defaults; and run a **thin** calendar Up-next spike
(read events + minimal home affordance + decision note) without enabling `--auto-start`.

## Why now

- **#3** — Deferred race on notes append; preempt before concurrent generate expands.
- **#4** — Opportunistic Still-open leftovers (not MLX resample / Swift XCTest / streaming).
- **#5** — v0.10.0 fast-follow still open; expands export breadth without changing default store.
- **thin #1** — Soak trigger fired; ROADMAP names #64 next — keep spike time-boxed and reversible.

## Scope note

Parents / links: [#64](https://github.com/Skalas/escriba/issues/64) (thin only); new issues for
append-notes / adapters / P2 as needed. **Out of scope:** full calendar auto-start,
bash install single-sourcing, Sparkle, MLX anti-alias resample, Swift XCTest target.

## Definition of Done

Done when: concurrent Enhance cannot drop an append (atomic server path + tests); 2–4 P2
items landed with ROADMAP updated; webhook + custom-script adapters work with env secrets /
argv-not-shell / stdlib HTTP and local-markdown remains default; thin calendar spike delivers
read + minimal Up-next (or documented blocker) plus a build-vs-park decision — **no**
`--auto-start` enablement unless product explicitly expands mid-sprint. Ship gate green.

## Seed test matrix

### Strand A — Atomic append-notes (#3) · HOLD

| ID | Criterion |
|----|-----------|
| T1 | Server endpoint (or DB helper) appends notes atomically (single transaction / compare-and-set) |
| T2 | Two concurrent Enhance/append calls on the same session both persist (no lost write) |
| T3 | SPA Enhance / generate continue to work; in-flight guard may remain as UX, not sole safety |
| T4 | Focused tests on the race |

### Strand B — P2 micro-bundle (#4) · REDUCE

| ID | Criterion |
|----|-----------|
| T5 | Triage Still open; pick 2–4 cheap items (prefer: notepad “Your notes” redundancy, `#31` SPA helpers / `SEEK_STEP_SECONDS`, narrow `except`, small config polish) |
| T6 | Each item has a test or no-test rationale |
| T7 | ROADMAP Still open updated |

### Strand C — Knowledge adapters (#5) · EXPAND

| ID | Criterion |
|----|-----------|
| T8 | `webhook` adapter: stdlib HTTP, secrets from env, failures degrade (don’t break session save) |
| T9 | `custom-script` adapter: argv list (no shell), env-configured path, timeout/fail-soft |
| T10 | Factory / config selects adapter; **local-markdown remains default** |
| T11 | Unit tests for both adapters (mocked HTTP / script) |

### Strand D — Thin calendar spike (#64) · EXPAND (thin)

| ID | Criterion |
|----|-----------|
| T12 | Reuse/extend `get_upcoming_events` for “today / soon” without new permission classes if possible |
| T13 | Minimal home “Up next” surface (one row or empty state) — spike UI, not full scheduling product |
| T14 | One-tap Record pre-titles session from event title when an event is selected (if events available) |
| T15 | Document Calendar permission gaps; **`--auto-start` stays blocked** with clear error |
| T16 | Spike decision note in ROADMAP / issue comment: build next vs park |

## Seed H-matrix (plan prose only — no `smoke.humanGates` in profile)

| ID | Type | What the human does |
|----|------|---------------------|
| H1 | ux | Approve thin Up-next home affordance (or reject / park) |
| H2 | live | On Mac: Calendar permission path works for reading today’s events |
| H3 | graduation | Product call: stay spike-only vs schedule full auto-start sprint later |

## Suggested issue mapping for prep

- Strand A → new issue (atomic append-notes)
- Strand B → chore / mini-checklist issue
- Strand C → new issue(s) under knowledge adapters
- Strand D → [#64](https://github.com/Skalas/escriba/issues/64) (thin scope in body)

## Explicit non-goals

- Enabling `watch-calendar --auto-start` or menubar calendar-driven start
- Closing entire Still-open backlog
- Checksum verification for release assets / install.sh Python rewrite
