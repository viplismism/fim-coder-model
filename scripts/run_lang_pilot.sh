#!/usr/bin/env bash
# One language, two-arm recipe (AST vs random), 6.7B: validation subset -> train AST -> train random -> validation scoring.
# One training run at a time; each arm is a fresh LoRA adapter from the same untouched base. Final evaluation
# (Real-FIM-Eval + locked held-out) is a separate step, run once after review.
# Usage (inside tmux): bash scripts/run_lang_pilot.sh <lang> <data_version> <config>
set -euo pipefail
cd "$(dirname "$0")/.."
LANG_=$1; DATA=$2; CFG=$3
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_SILENT=true
LOG=logs/$(date +%Y%m%dT%H%M%S%z)_${LANG_}_pilot_pipeline.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }
trap 'log "FAILED at: ${STEP:-startup} (exit $?)"' ERR
STEP=preflight
if pgrep -af -i "qwen|ollama|vllm" | grep -v pgrep | grep -q .; then log "a Qwen/LLM process is running"; exit 1; fi
if pgrep -f "python -m fimexp\.(train|score)" | grep -q .; then log "another training/scoring process is running"; exit 1; fi
$PY -m pytest -q tests 2>&1 | tail -1 | tee -a "$LOG"
test -f data/$DATA/validation_subset.json || $PY -m fimexp.val_subset --version $DATA --size 128 | tail -3 | tee -a "$LOG"
for arm in ast random; do
  STEP="train $arm"; log "$STEP (fresh adapter from the base)"
  $PY -m fimexp.train --config $CFG --data_version $DATA --arm $arm
  log "done: $(ls -dt logs/*_${LANG_}-${arm}_train_s42 | head -1)"
done
for name in base ast random; do
  STEP="validation scoring $name"; log "$STEP"
  AD=""; [ $name != base ] && AD="--adapter $(ls -dt logs/*_${LANG_}-${name}_train_s42 | head -1)/checkpoints/final_adapter"
  OUT=logs/$(date +%Y%m%dT%H%M%S%z)_${LANG_}-${name}_eval_validation
  $PY -m fimexp.score --config $CFG --tasks reth_ast,reth_random,reth_l2r --data_version $DATA --split validation $AD --out $OUT > $OUT.console.log 2>&1
  log "done: $OUT"
done
STEP=finished; log "pilot training + validation scoring finished; final evaluation waits for review"
