# Smoke — in-app updates + P2 + sidebar

| ID | Criterion | Result |
|----|-----------|--------|
| T1 | update-check API / service | PASS — `check-update --json`; tests/test_updates.py |
| T2 | Dashboard notify + dismiss | PASS — unit/UI wired; **human**: Settings → Check for updates, banner dismiss |
| T3 | One-click Update + progress | PASS — UpgradeService + tests; **human**: only on dirty-free install tree |
| T4 | Offline/CSRF fail-soft | PASS — tests + CSRF mutation guard |
| T5 | CLI check-update / update | PASS — exercised live |
| T6 | ROADMAP soak sync | PASS — docs in tree |
| T7 | Push main | PASS — done in prep |
| T8–T9 | P2 slice #105 | PASS — Range 416, mkstemp, prompt clear, export atomic + tests |
| T10–T11 | Sidebar clip | PASS — CSS in index.html; **human**: visual light/dark |

## Gaps for human UX
- Aesthetic: update banner + About card + sidebar titles under sticky dates
- Do not run live `escriba update` against a dirty tree / this sprint branch

## Note
GitHub `releases/latest` currently resolves to **v1.1.0** while app is **1.3.0** → `update_available: false`. Publish a GitHub Release for v1.3.0+ so in-app checks see newer tags.
