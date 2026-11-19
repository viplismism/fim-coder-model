#!/usr/bin/env bash
# After the repo-specific pipeline finishes (32B done): generate base and fine-tuned Qwen-14B completions on the 5%
# held-out validation files (random-hole and AST-hole sets), then score exact match ('their style', prvn.sh).
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PL=$(ls -t logs/*_repo_specific_pipeline.log | head -1); R=checks/repo_specific; D=data/rs-controlplane-qwen14b-v1; OUT=results/repo_specific/valstyle
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PL"; }
until grep -q "pipeline finished" $PL; do sleep 120; done
AD=$(ls -dt logs/*_qwen14b-random_train_s42 | head -1)/checkpoints/final_adapter
for arm in random ast; do
  for m in base ft; do
    log "val-style generation: qwen14b-$m on validation_$arm"
    .venv/bin/python $R/gen_eval.py generate configs/rs_qwen14b.yaml qwen14b-$m-val$arm $D/validation_$arm.jsonl $OUT $([ $m = ft ] && echo --adapter $AD) > logs/rs_valstyle_${m}_$arm.log 2>&1 || log "FAILED qwen14b-$m val $arm"
  done
done
python3 $R/val_style_eval.py $OUT $D results/repo_specific/valstyle_summary.json >> "$PL" 2>&1 && log "val-style check done"
