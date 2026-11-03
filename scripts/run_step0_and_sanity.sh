#!/usr/bin/env bash
# Chain: wait for the validation generations (step 0) -> in parallel:
#   CPU: cargo compile checks (120 validation pairs) -> structural report
#   GPU: 1.3B sanity check (checks/screening/PLAN.md section 0): train S0 (ast) and R0 (random) on rust-fim-v2,
#        then score base-1.3B / S0 / R0 on the full validation sets.
# VALIDATION ONLY. Test, benchmark and the locked held-out set are not touched.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_SILENT=true
GEN=results/val_structural/gen
LOG=results/val_structural/chain.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }

log "waiting for generation processes"
while pgrep -f "fimexp.gen_batch" >/dev/null; do sleep 30; done
for m in base ast random; do test "$(wc -l < $GEN/$m.jsonl)" -eq "$(wc -l < $GEN/items.jsonl)" || { log "generation for $m incomplete"; exit 1; }; done
log "generation complete"

(
  log "compile checks: start"
  $PY checks/diagnosis/compile_check.py $GEN 120 > results/val_structural/compile.log 2>&1
  log "compile checks: done"
  $PY checks/diagnosis/structural_report.py $GEN > results/val_structural/structural_report.txt 2>&1
  log "structural report written: results/val_structural/structural_report.txt"
) &
CPU=$!

(
  CFG=configs/sanity_1p3b.yaml
  for arm in ast random; do
    log "1.3B sanity: training $arm"
    $PY -m fimexp.train --config $CFG --data_version rust-fim-v2 --arm $arm > results/val_structural/sanity_train_$arm.log 2>&1
    log "1.3B sanity: training $arm done"
  done
  AST=$(ls -dt logs/*_1p3b-ast_train_s42 | head -1)/checkpoints/final_adapter
  RND=$(ls -dt logs/*_1p3b-random_train_s42 | head -1)/checkpoints/final_adapter
  for pair in "base:" "ast:$AST" "random:$RND"; do
    name=${pair%%:*}; ad=${pair#*:}
    log "1.3B sanity: scoring $name"
    $PY -m fimexp.score --config $CFG --tasks reth_ast,reth_random,reth_l2r --data_version rust-fim-v2 --split validation \
        ${ad:+--adapter "$ad"} --out logs/$(date +%Y%m%dT%H%M%S%z)_1p3b-${name}_eval_validation > results/val_structural/sanity_score_$name.log 2>&1
  done
  log "1.3B sanity: all done"
) &
GPU=$!
wait $CPU $GPU
log "chain finished"
