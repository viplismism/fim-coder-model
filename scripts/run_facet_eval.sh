#!/usr/bin/env bash
# Facet evaluation queue. Waits until BOTH fine-tunes have a final adapter and no training process is left, then:
#  GPU (one job at a time): greedy completions on the test set for base / AST / random (sensitive examples first,
#      so pass@1 scoring can start early; then the rest for exact match), then perplexity for all three, then the
#      validation-file ('their style') completions on AST-hole and random-hole validation sets.
#  CPU (alongside the GPU): cargo build + test scoring of each model's completions as soon as its sensitive set is
#      generated (3 workers, own target dirs); exact-match scoring of the rest at the end.
#  Then: summary (results/repo_specific/facet/summary.json), val-style summary, HTML cards (gitignored).
# Every step is resumable: rerun this script after a crash or reboot and finished work is skipped.
# Never tunes anything on test results. Qwen-32B is NOT run here (only if a fine-tune beats base; decided after).
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
R=checks/repo_specific; CFG=configs/rs_facet_qwen14b.yaml; D=data/rs-facet-qwen14b-v1
OUT=results/repo_specific/facet; VAL=$OUT/valstyle; REPO=sources_repo_specific/facet-rs__facet
PL=logs/facet_eval_pipeline.log; PY=.venv/bin/python
mkdir -p $OUT $VAL
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PL"; }

adapter() { ls -d logs/*_facet-qwen14b-$1_train_s42/checkpoints/final_adapter 2>/dev/null | sort | tail -1; }
until [ -n "$(adapter ast)" ] && [ -n "$(adapter random)" ] && ! pgrep -f "fimexp/train.py" >/dev/null && ! pgrep -f "fimexp.train" >/dev/null; do
  sleep 300
done
AST=$(adapter ast); RND=$(adapter random)
log "eval start; adapters: $AST | $RND"

# sensitive-first split of the 1,817 usable examples (deterministic)
[ -f $OUT/eval_sensitive.jsonl ] || python3 - <<E
import json
ex=[json.loads(l) for l in open("$R/facet_eval.jsonl")]
open("$OUT/eval_sensitive.jsonl","w").write("".join(json.dumps(e)+"\n" for e in ex if e["sensitive"]))
open("$OUT/eval_rest.jsonl","w").write("".join(json.dumps(e)+"\n" for e in ex if not e["sensitive"]))
E

declare -A AD=([qwen14b-base]="" [facet-ast]="--adapter $AST" [facet-random]="--adapter $RND")
MODELS="qwen14b-base facet-ast facet-random"
SCORE_PIDS=()
gen() {  # model output_name examples out_dir
  log "generate $2 on $(basename $3)"
  $PY $R/gen_eval.py generate $CFG $2 $3 $4 ${AD[$1]} >> logs/facet_gen_$2.log 2>&1 || log "FAILED generate $2 $3"
}
score() {  # name
  $PY $R/gen_eval.py score $REPO $R/facet_eval.jsonl $OUT/$1.completions.jsonl $OUT/$1.scores.jsonl 3 --lang rust >> logs/facet_score_$1.log 2>&1 \
    && log "scored $1 ($(wc -l < $OUT/$1.scores.jsonl) rows)" || log "FAILED score $1"
}

# 1. sensitive set, each model; cargo scoring of a model runs while the next model generates (one scorer at a time)
for m in $MODELS; do
  gen $m $m $OUT/eval_sensitive.jsonl $OUT
  [ ${#SCORE_PIDS[@]} -gt 0 ] && wait ${SCORE_PIDS[@]}
  score $m & SCORE_PIDS=($!)
done
# 2. the rest (exact match only)
for m in $MODELS; do gen $m $m $OUT/eval_rest.jsonl $OUT; done
# 3. perplexity of the real code
for m in $MODELS; do
  log "ppl $m"; $PY $R/gen_eval.py ppl $CFG $m $R/facet_eval.jsonl $OUT ${AD[$m]} >> logs/facet_ppl_$m.log 2>&1 || log "FAILED ppl $m"
done
# 4. validation files, their style
for arm in ast random; do for m in $MODELS; do gen $m $m-val$arm $D/validation_$arm.jsonl $VAL; done; done
wait ${SCORE_PIDS[@]}
# 5. exact-match scoring of the rest (no cargo needed for non-sensitive rows; resumable, skips scored ids)
for m in $MODELS; do score $m; done
# 6. summaries + cards
$PY $R/summarize_facet.py $OUT $OUT/summary.json >> "$PL" 2>&1 && log "summary written"
$PY $R/val_style_eval_facet.py $VAL $D $OUT/valstyle_summary.json >> "$PL" 2>&1 && log "val-style summary written"
$PY $R/make_compare_html_facet.py $OUT $R/facet_eval.jsonl $OUT/summary.json results/repo_specific/compare_facet.html >> "$PL" 2>&1 && log "html written"
log "facet eval pipeline finished"
