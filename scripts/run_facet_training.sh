#!/usr/bin/env bash
# facet repo-specific: two Qwen-14B LoRA runs with identical settings, AST then random (rs-facet-qwen14b-v1).
# If a run's process dies (not a reboot), it is resumed from its latest checkpoint (up to 3 times).
# After a reboot: rerun this script; it resumes an unfinished run and skips finished ones.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_SILENT=true
LOG=logs/facet_training_pipeline.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }
T="--config configs/rs_facet_qwen14b.yaml --data_version rs-facet-qwen14b-v1"
for arm in ast random; do
  last=$(ls -d logs/*_facet-qwen14b-${arm}_train_s42 2>/dev/null | sort | tail -1)
  if [ -n "$last" ] && [ -d "$last/checkpoints/final_adapter" ]; then log "$arm: already finished ($last)"; continue; fi
  if [ -n "$last" ] && ls -d $last/checkpoints/checkpoint-* >/dev/null 2>&1; then
    log "$arm: resuming $last"; .venv/bin/python -m fimexp.train $T --arm $arm --resume_run_dir $last >> logs/facet_train_$arm.log 2>&1
  else
    log "$arm: starting fresh"; .venv/bin/python -m fimexp.train $T --arm $arm >> logs/facet_train_$arm.log 2>&1
  fi
  for try in 1 2 3; do
    last=$(ls -d logs/*_facet-qwen14b-${arm}_train_s42 | sort | tail -1)
    [ -d "$last/checkpoints/final_adapter" ] && break
    log "$arm: process ended without final adapter; resume attempt $try"
    .venv/bin/python -m fimexp.train $T --arm $arm --resume_run_dir $last >> logs/facet_train_$arm.log 2>&1
  done
  [ -d "$last/checkpoints/final_adapter" ] && log "$arm: finished ($last)" || { log "$arm: FAILED after retries"; exit 1; }
done
log "both runs finished"
