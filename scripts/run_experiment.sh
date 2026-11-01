#!/usr/bin/env bash
# Full experiment pipeline, sequential (measured faster than running the arms concurrently).
# Run inside tmux:  tmux new -s fim 'bash scripts/run_experiment.sh'     (attach later: tmux attach -t fim)
#   --preflight   checks only; nothing is trained, scored or deleted
#
# Steps: preflight -> discard smoke adapters -> base validation scoring -> AST train -> AST validation scoring
#        -> random train -> random validation scoring. Stops at the first failure.
# Every step writes its own logs/<timestamp>_... folder; this script's own log is logs/<timestamp>_pipeline.log.
# Test split and benchmark are NOT run here (final evaluation only).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=rust-fim-v2
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_SILENT=true
STAMP=$(date +%Y%m%dT%H%M%S%z)
mkdir -p logs
PIPE_LOG=logs/${STAMP}_pipeline.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PIPE_LOG"; }
trap 'log "FAILED at: ${STEP:-startup} (exit $?)"' ERR

STEP=preflight
log "preflight"
if pgrep -af -i "qwen|ollama|vllm" | grep -v pgrep | grep -q .; then log "a Qwen/LLM process is running; stop it first"; exit 1; fi
if pgrep -f "fimexp.train --data_version|fimexp.score --tasks" | grep -q .; then log "another training/scoring process is running"; exit 1; fi
$PY - <<'EOF' | tee -a "$PIPE_LOG"
import sys; sys.path.insert(0, ".")
from fimexp import sources, wandb_log
from fimexp.train import missing_settings
cfg = sources.load_config()
print("pins:", sources.verify_pins(cfg))
assert not missing_settings(cfg), missing_settings(cfg)
assert cfg["base_precision"] == "bf16" and cfg["training"]["optim"] == "adamw_torch" and cfg["training"]["seed"] == 42
w = cfg["wandb"]
if w["enabled"]:
    acc = wandb_log.project_access(w["entity"], w["project"])
    print(f"wandb {w['entity']}/{w['project']} access: {acc}")
    assert acc in w["allowed_access"], "W&B project is not private"
print("settings:", {k: cfg["training"][k] for k in ("num_train_epochs", "per_device_batch_size", "gradient_accumulation_steps",
      "learning_rate", "lr_scheduler_type", "warmup_ratio", "weight_decay", "max_grad_norm", "optim", "seed", "eval_steps", "save_steps")},
      "lora:", cfg["lora"])
EOF
$PY -m pytest -q tests 2>&1 | tail -1 | tee -a "$PIPE_LOG"
free -g | sed -n 2p | tee -a "$PIPE_LOG"
SMOKE_ADAPTERS=$(ls -d logs/*_smoke_s42/checkpoints 2>/dev/null || true)
log "smoke adapters to discard before training: ${SMOKE_ADAPTERS:-none}"
if [ "${1:-}" = "--preflight" ]; then log "preflight only: done"; exit 0; fi

STEP="discard smoke adapters"
for d in $SMOKE_ADAPTERS; do rm -rf "$d"; log "removed $d (run logs kept)"; done

score() {  # $1 = name, $2 = optional adapter path
  STEP="validation scoring: $1"; log "$STEP"
  OUT=logs/$(date +%Y%m%dT%H%M%S%z)_${1}_eval_validation
  $PY -m fimexp.score --tasks reth_ast,reth_random,reth_l2r --data_version $DATA --split validation \
      ${2:+--adapter "$2"} --out "$OUT" 2>&1 | tee "$OUT.console.log"
  log "done: $OUT"
}
train() {  # $1 = arm
  STEP="training: $1"; log "$STEP"
  $PY -m fimexp.train --data_version $DATA --arm "$1"
  RUN=$(ls -dt logs/*_"$1"_train_s42 | head -1)
  test -d "$RUN/checkpoints/final_adapter"
  log "done: $RUN"
}

score base
train ast;    score ast "$(ls -dt logs/*_ast_train_s42 | head -1)/checkpoints/final_adapter"
train random; score random "$(ls -dt logs/*_random_train_s42 | head -1)/checkpoints/final_adapter"
STEP=finished; log "all steps finished"
