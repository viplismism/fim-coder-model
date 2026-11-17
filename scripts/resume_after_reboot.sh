#!/usr/bin/env bash
# Resume after the unplanned reboot of 2026-11-16 10:29: the C++ final evaluation had finished base (complete, kept) and
# was partway through ast (discarded, rescored from scratch). Same scorer and commands as run_lang_final_eval.sh.
# Then the Go final evaluation (full) and both summaries + pushes. One GPU job at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
LOG=logs/$(date +%Y%m%dT%H%M%S%z)_cpp_final_eval_pipeline.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }
AST=$(ls -dt logs/*_cpp-ast_train_s42 | head -1)/checkpoints/final_adapter
RND=$(ls -dt logs/*_cpp-random_train_s42 | head -1)/checkpoints/final_adapter
log "resume after reboot: adapters ast=$AST random=$RND | held-out cpp-fim-v1-heldout (manifest sha256 $(sha256sum data/cpp-fim-v1-heldout/manifest.json | cut -c1-16))"
log "done: logs/20261116T074350+0530_cpp-base_eval_final (scored before the reboot, complete)"
ok=1
for pair in "ast:$AST" "random:$RND"; do
  name=${pair%%:*}; ad=${pair#*:}
  log "final scoring $name"
  OUT=logs/$(date +%Y%m%dT%H%M%S%z)_cpp-${name}_eval_final
  if $PY -m fimexp.score --config configs/lang_cpp.yaml --lang cpp --tasks bench_add,bench_edit,heldout_ast,heldout_random,heldout_l2r \
      --final_heldout --heldout_version cpp-fim-v1-heldout --adapter "$ad" --out "$OUT" > "$OUT.console.log" 2>&1; then
    log "done: $OUT"
  else log "FAILED at: final scoring $name"; ok=0; break; fi
done
[ $ok = 1 ] && log "final evaluation finished" && bash scripts/after_lang_final_eval.sh cpp "$LOG"
bash scripts/run_lang_final_eval.sh go configs/lang_go.yaml go-fim-v1-heldout
bash scripts/after_lang_final_eval.sh go "$(ls -t logs/*_go_final_eval_pipeline.log | head -1)"
