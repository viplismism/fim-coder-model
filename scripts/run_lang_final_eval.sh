#!/usr/bin/env bash
# ONE-TIME final evaluation for one language: base, AST and random (6.7B, seed 42) on the language's Real-FIM-Eval subset
# (Add = primary score; Edit = provisional layout) and its locked held-out set. Same scorer and rules as Rust.
# Usage (inside tmux): bash scripts/run_lang_final_eval.sh <lang> <config> <heldout_version>
set -euo pipefail
cd "$(dirname "$0")/.."
LANG_=$1; CFG=$2; HELD=$3
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
LOG=logs/$(date +%Y%m%dT%H%M%S%z)_${LANG_}_final_eval_pipeline.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }
trap 'log "FAILED at: ${STEP:-startup} (exit $?)"' ERR
STEP=preflight
if pgrep -f "python -m fimexp\.(train|score)" | grep -q .; then log "another training/scoring process is running"; exit 1; fi
AST=$(ls -dt logs/*_${LANG_}-ast_train_s42 | head -1)/checkpoints/final_adapter
RND=$(ls -dt logs/*_${LANG_}-random_train_s42 | head -1)/checkpoints/final_adapter
test -d "$AST" && test -d "$RND"
log "adapters: ast=$AST random=$RND | held-out $HELD (manifest sha256 $(sha256sum data/$HELD/manifest.json | cut -c1-16))"
for pair in "base:" "ast:$AST" "random:$RND"; do
  name=${pair%%:*}; ad=${pair#*:}
  STEP="final scoring $name"; log "$STEP"
  OUT=logs/$(date +%Y%m%dT%H%M%S%z)_${LANG_}-${name}_eval_final
  $PY -m fimexp.score --config $CFG --lang $LANG_ --tasks bench_add,bench_edit,heldout_ast,heldout_random,heldout_l2r \
      --final_heldout --heldout_version $HELD ${ad:+--adapter "$ad"} --out "$OUT" > "$OUT.console.log" 2>&1
  log "done: $OUT"
done
STEP=finished; log "final evaluation finished"
