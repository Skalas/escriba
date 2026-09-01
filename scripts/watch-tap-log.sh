#!/usr/bin/env bash
# Watch the Swift audio bridge's own log lines in real time.
# Used by .metate/t12-listener-check.md (issue #217) — the A2DP→HFP gate.
#
# Every line the bridge writes to stderr is drained into app.log by
# screen_capture.py::_drain_stderr and prefixed "[swift] [tap]".
set -euo pipefail

LOG="${ESCRIBA_LOG:-$HOME/Library/Logs/escriba/app.log}"

if [[ ! -f "$LOG" ]]; then
  echo "No log at $LOG — is Escriba running?" >&2
  exit 1
fi

echo "Watching $LOG for bridge activity. Ctrl-C to stop."
echo "Looking for: 'rebuilt after route/format change' (T12 pass signal)"
echo
# --line-buffered so grep does not hold lines back in a pipe.
tail -F "$LOG" | grep --line-buffered -E "\[tap\]|\[swift\]|degraded|silent for" \
  | while IFS= read -r line; do
      case "$line" in
        *"rebuilt after route/format change"*) printf '\033[32m%s\033[0m\n' "$line" ;;
        *"rebuilt after IO proc stall"*)        printf '\033[33m%s\033[0m\n' "$line" ;;
        *"[tap] dead"*|*"failed"*)             printf '\033[31m%s\033[0m\n' "$line" ;;
        *"skipped rebuild"*)                   printf '\033[36m%s\033[0m\n' "$line" ;;
        *) printf '%s\n' "$line" ;;
      esac
    done
