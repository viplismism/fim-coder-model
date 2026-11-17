#!/usr/bin/env bash
# GPU queue: for each language in order, wait for its prep gate; if READY run the two-arm pilot (train AST, train random,
# validation scoring) and then the one-time final evaluation, then summarise + commit/push the summary. If STOP, skip
# the language (its reasons are in checks/multilang/<lang>_gate.json). One GPU job at a time.
# Usage: bash scripts/run_lang_queue.sh typescript go cpp
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/$(date +%Y%m%dT%H%M%S%z)_lang_queue.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }
for L in "$@"; do
  G=checks/multilang/${L}_gate.json
  log "$L: waiting for prep gate"
  until [ -f $G ]; do sleep 120; done
  S=$(python3 -c "import json; print(json.load(open('$G'))['status'])")
  if [ "$S" != READY ]; then log "$L: gate $S, skipped: $(python3 -c "import json; print(json.load(open('$G'))['problems'])")"; continue; fi
  DATA=$(python3 -c "import json; print(json.load(open('$G'))['data_version'])"); HELD=$(python3 -c "import json; print(json.load(open('$G'))['heldout_version'])")
  log "$L: READY ($DATA, held-out $HELD); pilot"
  bash scripts/run_lang_pilot.sh $L $DATA configs/lang_$L.yaml || { log "$L: pilot FAILED; skipping to next language"; continue; }
  log "$L: final evaluation"
  bash scripts/run_lang_final_eval.sh $L configs/lang_$L.yaml $HELD
  PIPE=$(ls -t logs/*_${L}_final_eval_pipeline.log | head -1)
  bash scripts/after_lang_final_eval.sh $L $PIPE || log "$L: summary/push step failed (see logs/after_${L}_final_eval.log)"
  log "$L: done"
done
log "queue finished"
