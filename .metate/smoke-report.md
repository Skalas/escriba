# Smoke — verify updates + install parity + P2 + docs

| ID | Criterion | Result |
|----|-----------|--------|
| T1 | Newer release tag available | PASS at ship — cut `v1.3.1` GitHub Release |
| T2 | Older current → update_available | PASS — `ESCRIBA_VERSION_OVERRIDE=1.2.0` / `--current 1.2.0` → true vs `v1.3.0` (pre-1.3.1) |
| T3 | Settings Check for updates UI | **human** — About button; banner when available |
| T4 | Guarded install | PASS — soak=False on mutating path; dirty-tree refuse unchanged |
| T5 | Matching version → up-to-date | PASS after install at 1.3.1 / override cleared |
| T6–T8 | Install path inventory | PASS — `install_paths.py` + tests; intentional diffs documented |
| T9–T10 | #105 slice | PASS — watch bound, LLM delimiter, mix lengths, TranscriptionError |
| T11–T12 | Docs | PASS — CHANGELOG `[1.3.1]`, ROADMAP |

## Soak recipe
```bash
# Pretend older build (check-only; does not drive install)
ESCRIBA_VERSION_OVERRIDE=1.3.0 uv run escriba check-update --json
# After v1.3.1 is Latest → update_available: true

# Or without .env:
uv run escriba check-update --current 1.3.0 --json
```

## Human UX
Settings → About → Check for updates (and banner if available).
