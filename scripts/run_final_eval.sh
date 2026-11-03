#!/usr/bin/env bash
# Final evaluation (run once, after training is finished and settings are frozen):
# base, AST-trained and random-trained models on the held-out TEST split (AST holes, random holes,
# left-to-right) and on Real-FIM-Eval Rust (Add; Edit is PROVISIONAL: inferred prompt layout).
# Same examples and scoring rules for every model; nothing here selects settings or checkpoints.
# Run inside tmux:  tmux new -s fim-final 'bash scripts/run_final_eval.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=rust-fim-v2
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
STAMP=$(date +%Y%m%dT%H%M%S%z)
PIPE_LOG=logs/${STAMP}_final_eval_pipeline.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PIPE_LOG"; }
trap 'log "FAILED at: ${STEP:-startup} (exit $?)"' ERR

STEP=preflight
if pgrep -af -i "qwen|ollama|vllm" | grep -v pgrep | grep -q .; then log "a Qwen/LLM process is running; stop it first"; exit 1; fi
if pgrep -f "fimexp.train --data_version|fimexp.score --tasks" | grep -q .; then log "another training/scoring process is running"; exit 1; fi
AST=$(ls -dt logs/*_ast_train_s42 | head -1)/checkpoints/final_adapter
RND=$(ls -dt logs/*_random_train_s42 | head -1)/checkpoints/final_adapter
test -d "$AST" && test -d "$RND"
log "adapters: ast=$AST random=$RND"
$PY -c "import sys; sys.path.insert(0, '.'); from fimexp import sources; print(sources.verify_pins(sources.load_config()))" | tee -a "$PIPE_LOG"

score() {  # $1 = name, $2 = optional adapter
  STEP="final scoring: $1"; log "$STEP"
  OUT=logs/$(date +%Y%m%dT%H%M%S%z)_${1}_eval_final
  $PY -m fimexp.score --tasks reth_ast,reth_random,reth_l2r,bench_add,bench_edit --data_version $DATA \
      --split test --final ${2:+--adapter "$2"} --out "$OUT" 2>&1 | tee "$OUT.console.log"
  log "done: $OUT"
}
score base
score ast "$AST"
score random "$RND"
STEP=finished; log "final evaluation finished"
