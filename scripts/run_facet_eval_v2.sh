#!/usr/bin/env bash
# Facet evaluation queue, REORDERED 2026-11-23 at the user's request (replaces the remaining steps of
# run_facet_eval.sh and run_facet_eval_32b.sh; everything already generated/scored is reused, all steps resumable):
#   1. exact-match completions on the non-sensitive rest for AST and random (base already done)
#   2. base Qwen2.5-Coder-32B (no fine-tune) on the sensitive set: completions, then cargo build+test scoring
#      (pass@1 + exact match + stop-limit check), as a 4th column
#   3. perplexity of the real code for base-14B / AST / random
#   4. 'their style' check on a fixed random sample of 150 validation holes per hole type (seed 20261123,
#      same sample for every model; the samples are written once and reused)
#   5. exact-match scoring of the rest, final summary (with --extra qwen32b-base), val-style summary, HTML cards
# One GPU job at a time; CPU scoring runs alongside generation. Nothing is tuned on test results. No training.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
R=checks/repo_specific; CFG=configs/rs_facet_qwen14b.yaml; CFG32=configs/rs_facet_qwen32b.yaml; D=data/rs-facet-qwen14b-v1
OUT=results/repo_specific/facet; VAL=$OUT/valstyle; REPO=sources_repo_specific/facet-rs__facet
PL=logs/facet_eval_pipeline.log; PY=.venv/bin/python; M32=qwen32b-base
mkdir -p $OUT $VAL
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PL"; }
adapter() { ls -d logs/*_facet-qwen14b-$1_train_s42/checkpoints/final_adapter 2>/dev/null | sort | tail -1; }
AST=$(adapter ast); RND=$(adapter random)
declare -A AD=([qwen14b-base]="" [facet-ast]="--adapter $AST" [facet-random]="--adapter $RND")
MODELS="qwen14b-base facet-ast facet-random"
gen() {  # model output_name examples out_dir [config]
  log "generate $2 on $(basename $3)"
  $PY $R/gen_eval.py generate ${5:-$CFG} $2 $3 $4 ${AD[$1]} >> logs/facet_gen_$2.log 2>&1 || log "FAILED generate $2 $3"
}
score() {  # name
  $PY $R/gen_eval.py score $REPO $R/facet_eval.jsonl $OUT/$1.completions.jsonl $OUT/$1.scores.jsonl 3 --lang rust >> logs/facet_score_$1.log 2>&1 \
    && log "scored $1 ($(wc -l < $OUT/$1.scores.jsonl) rows)" || log "FAILED score $1"
}

# wait for the generation started by the old queue (AST rest) to exit; it resumes below if it was interrupted
while pgrep -f "gen_eval.py generate" >/dev/null; do sleep 60; done
log "queue v2 (reordered): rest -> 32B -> ppl -> style check (150/hole type) -> summaries"

# 1. rest, exact match only (base done; generate skips finished ids)
for m in facet-ast facet-random; do gen $m $m $OUT/eval_rest.jsonl $OUT; done
# exact-match scoring of the rest for the three 14B models, on CPU while the 32B generates
( for m in $MODELS; do score $m; done ) & RESTSCORE=$!
# 2. base 32B on the sensitive set, then cargo scoring
AD[$M32]=""
gen $M32 $M32 $OUT/eval_sensitive.jsonl $OUT $CFG32
wait $RESTSCORE
score $M32
# 3. perplexity
for m in $MODELS; do
  log "ppl $m"; $PY $R/gen_eval.py ppl $CFG $m $R/facet_eval.jsonl $OUT ${AD[$m]} >> logs/facet_ppl_$m.log 2>&1 || log "FAILED ppl $m"
done
# 4. style check on a fixed 150-hole sample per hole type
for arm in ast random; do
  [ -f $VAL/sample150_$arm.jsonl ] || python3 - <<E
import json, random
ex = [l for l in open("$D/validation_$arm.jsonl")]
idx = sorted(random.Random(20261123).sample(range(len(ex)), 150))
open("$VAL/sample150_$arm.jsonl", "w").write("".join(ex[i] for i in idx))
E
  for m in $MODELS; do gen $m $m-val$arm $VAL/sample150_$arm.jsonl $VAL; done
done
# 5. summaries + cards
$PY $R/summarize_facet.py $OUT $OUT/summary.json --extra $M32 >> "$PL" 2>&1 && log "summary written (4 columns)"
$PY $R/val_style_eval_facet.py $VAL $D $OUT/valstyle_summary.json >> "$PL" 2>&1 && log "val-style summary written (150-hole samples)"
$PY $R/make_compare_html_facet.py $OUT $R/facet_eval.jsonl $OUT/summary.json results/repo_specific/compare_facet.html >> "$PL" 2>&1 && log "html written"
log "facet eval pipeline finished"
