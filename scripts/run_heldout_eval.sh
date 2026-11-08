#!/usr/bin/env bash
# ONE-TIME evaluation of the locked fresh held-out set (data/heldout-v1: 15 post-2024 Rust repos) with the final
# 6.7B models: base, AST-trained and random-trained (seed 42). Same scorer and rules as the other evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
LOG=logs/$(date +%Y%m%dT%H%M%S%z)_heldout_eval_pipeline.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }
AST=logs/20261101T105542+0530_ast_train_s42/checkpoints/final_adapter
RND=logs/20261101T230132+0530_random_train_s42/checkpoints/final_adapter
test -d "$AST" && test -d "$RND"
log "held-out set manifest sha256: $(sha256sum data/heldout-v1/manifest.json | cut -c1-16)"
for pair in "base:" "ast:$AST" "random:$RND"; do
  name=${pair%%:*}; ad=${pair#*:}
  log "scoring $name"
  OUT=logs/$(date +%Y%m%dT%H%M%S%z)_${name}_eval_heldout
  $PY -m fimexp.score --tasks heldout_ast,heldout_random,heldout_l2r --final_heldout ${ad:+--adapter "$ad"} --out "$OUT" > "$OUT.console.log" 2>&1
  log "done: $OUT"
done
log "held-out evaluation finished"
