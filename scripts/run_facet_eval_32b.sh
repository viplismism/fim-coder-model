#!/usr/bin/env bash
# Qwen2.5-Coder-32B BASE (no fine-tune) on the facet test set, as a 4th column next to base-14B / AST / random.
# Approved by the user 2026-11-23, after both 14B fine-tunes beat base. Waits until the main facet eval pipeline
# (run_facet_eval.sh) has finished and no generation/scoring process is left (one GPU job at a time), then:
#   greedy completions on the SAME sensitive examples (configs/rs_facet_qwen32b.yaml differs from the 14B config
#   only in the model), cargo build+test scoring (3 workers), and the final summary with --extra qwen32b-base.
# Resumable: rerun after a crash and finished work is skipped. Evaluation only, no training, nothing tuned on test.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
R=checks/repo_specific; OUT=results/repo_specific/facet; REPO=sources_repo_specific/facet-rs__facet
PL=logs/facet_eval_pipeline.log; PY=.venv/bin/python; M=qwen32b-base
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PL"; }

until grep -q "facet eval pipeline finished" "$PL" && ! pgrep -f "gen_eval.py" >/dev/null && ! pgrep -f "run_facet_eval.sh" >/dev/null; do
  sleep 300
done
log "32b: generate $M on eval_sensitive.jsonl"
$PY $R/gen_eval.py generate configs/rs_facet_qwen32b.yaml $M $OUT/eval_sensitive.jsonl $OUT >> logs/facet_gen_$M.log 2>&1 \
  || { log "32b: FAILED generate"; exit 1; }
log "32b: score $M"
$PY $R/gen_eval.py score $REPO $R/facet_eval.jsonl $OUT/$M.completions.jsonl $OUT/$M.scores.jsonl 3 --lang rust >> logs/facet_score_$M.log 2>&1 \
  && log "32b: scored $M ($(wc -l < $OUT/$M.scores.jsonl) rows)" || { log "32b: FAILED score"; exit 1; }
$PY $R/summarize_facet.py $OUT $OUT/summary.json --extra $M >> "$PL" 2>&1 && log "32b: summary written with 4th column"
log "32b: finished"
