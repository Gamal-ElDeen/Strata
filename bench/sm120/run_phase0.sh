#!/bin/bash
# Sequential Phase 0 cells. One engine at a time. Stops the queue on a
# non-zero cell only when the GPU is left busy; otherwise records the failure
# and continues. Prints one OK or FAILED line per cell, then ALL_DONE.
set -u
PY=/home/righteous/ai/Strata/.venv/bin/python
H=/home/righteous/ai/Strata-decode/bench/sm120/measure_decode.py
ART=/home/righteous/ai/dflash2-artifacts/phase0
PR=/home/righteous/ai/Strata-pr3322/build-dflash2/strata
GGUF=/home/righteous/ai/dflash2-artifacts/dflash2-bf16.gguf
mkdir -p "$ART"
LOG=$ART/driver.log
: > "$LOG"

run() {
  local arm=$1
  shift
  echo "START $arm $(date -u +%H:%M:%S)" >> "$LOG"
  if "$PY" "$H" "$arm" "$@" >> "$LOG" 2>&1; then
    echo "OK $arm"
    echo "OK $arm" >> "$LOG"
  else
    echo "FAILED $arm exit=$?"
    echo "FAILED $arm" >> "$LOG"
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
      echo "FAILED gpu still busy after $arm"
      exit 1
    fi
  fi
}

run mtp-base
run mtp-spec3 -- --spec 3
run mtp-spec5 -- --spec 5
run mtp-spec6 -- --spec 6
run mtp-minp03 -- --spec-min-p 0.3
run mtp-minp07 -- --spec-min-p 0.7
run mtp-pool13 -- --pool-workers 13
run mtp-suffix0 -- --suffix-draft 0
run mtp-profile --profile --repeats 1
run dflash-k3-suffix0 --binary "$PR" --no-mtp -- --dflash "$GGUF" --dflash-window 0 --dflash-block 3 --suffix-draft 0
echo "ALL_DONE"
echo "ALL_DONE" >> "$LOG"
