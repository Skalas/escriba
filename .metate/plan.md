# Sprint plan — Harden & shrink the core (pre-1.0)

> **Mode hint:** REDUCE → HOLD. Structural only — no new features. Shrink the
> surface (dead code + layer debt), then refactor the central POST dispatcher on
> the smaller surface. All core-loop changes ship with tests.
>
> **Why this, why now:** The roadmap's own sequencing is "harden the core before
> we widen, depth before breadth." v1.0.0 is the declared next milestone but its
> premise is a *frozen* spine proven by a real-meeting soak. This sprint cleans and
> hardens that spine first so the 1.0 soak is meaningful. Next sprint after this is
> `v1.0.0` release hardening (the roadmap "next up" item — deferred here on purpose).
>
> **Source signals:** graph hotspot (`do_POST` complexity), graph 0-callers (dead
> code), v0.10.0 deferred design debt (layer inversion). Calendar spike (#64)
> explicitly deferred — it widens scope, against pre-1.0 depth-before-breadth.

---

## Candidate #2 — `do_POST` god-function refactor + route-parse hardening  [HOLD]

`do_POST` in `app/server.py` is the single dispatch point for every mutating
endpoint: cognitive complexity ~55, ~23 callees, 16 `elif` arms. Route parsing
uses `split`/`rsplit` with no validation, so malformed paths reach handler logic.
Core-loop file → ships with tests; behavior must stay identical.

**why now:** graph hotspot; single point of failure for all POST mutations.
**blast-radius:** every mutating endpoint (start/stop, split/merge, move, export,
retranscribe, generate-notes, enhance-prompt, folder CRUD).

### DoD / test matrix
- **T1** — `do_POST` decomposed into per-route dispatch (lookup table or
  per-route methods); each route handler isolated so one route's error can't
  cascade. Behavior unchanged across all existing POST endpoints.
- **T2** — Malformed/ambiguous paths (`/api/sessions//split`,
  `/api/sessions/<id>/split/extra`, trailing slashes, non-numeric ids) return
  `404`/`400` with structured errors — never a stack trace or wrong-route hit.
- **T3** — The existing route-level server tests (24 in `test_server.py`) stay
  green; new tests cover the malformed-path cases in T2.

---

## Candidate #3 — Pay down v0.10.0 layer-inversion + adapter design debt  [REDUCE]

Deferred at the v0.10.0 close. The `local-markdown` knowledge adapter imports the
Markdown formatter from `app/server.py` (infrastructure → presentation inversion);
provider dispatch is a bare string check; there's a dead inner `except` in
`_build_custom_prompt`; `_make_handler` is duplicated instead of living in
`tests/conftest.py`.

**why now:** explicitly deferred; clean-architecture violation flagged at sprint
close. **blast-radius:** low — relocate a formatter, swap a string check for a
factory, dedup a test helper.

### DoD / test matrix
- **T1** — Markdown formatter moved to a shared module; the `local-markdown`
  adapter no longer imports from `app/server.py` (no infra→presentation import).
- **T2** — Knowledge-store provider selection routes through a factory rather than
  a bare string comparison; default stays `local-markdown`, fully local.
- **T3** — Dead inner `except` in `_build_custom_prompt` removed; `_make_handler`
  deduped into `tests/conftest.py`. Full suite green.

---

## Candidate #4 — Remove dead code (MPS backend + daemon signal handler)  [REDUCE]

Graph reports 0 callers for `StreamingTranscriberMPS`
(`transcribe/streaming_mps.py`, ~249 LOC, deprecated — its own docstring warns of
MPS instability) and for the unused `_signal_handler` in `DaemonServer`.

**why now:** dead paths add maintenance burden with zero active usage.
**blast-radius:** low — but confirm no dynamic/config reference selects MPS before
deleting.

### DoD / test matrix
- **T1** — MPS streaming path removed (or moved to `legacy/`); no broken imports,
  and no `escriba.toml`/config/CLI key still selects `mps` as a backend (or it
  degrades gracefully with a clear message).
- **T2** — `_signal_handler` removed from `DaemonServer`; daemon start/stop and
  shutdown behavior unaffected.
- **T3** — Full suite green; `uv run ruff check .` and `uv run mypy .` clean.

---

## Sprint Definition of Done

Done when: the POST dispatcher is decomposed and malformed paths are rejected
(T2.1–T2.3), the layer inversion + adapter debt is paid down (T3.1–T3.3), the dead
MPS/daemon code is gone (T4.1–T4.3), and the full ship gate
(`ruff` + `mypy` + `pytest`) is green. No behavior regressions on existing
endpoints. Next sprint: `v1.0.0` release hardening (roadmap "next up").
