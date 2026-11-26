#!/usr/bin/env bash
# facet-rs/facet (Rust), cutoff 2026-01-01: the main result. Qwen2.5-Coder-14B + LoRA (random holes, and an AST-hole
# control with identical settings) vs base 14B and base 32B, scored by cargo build + tests on real post-cutoff edits.
#   bash scripts/facet.sh [testset|data|train|eval|all]      (default: all; every stage is resumable)
# Needs: scripts/setup.sh, `scripts/fetch.sh facet`, rustup. GPU: ~40 GB for 14B training, ~70 GB for 32B inference.
# Time on one DGX Spark: test set ~hours (CPU, cargo), data ~3 min, training ~25 h per arm, eval ~1 day.
LOG_NAME=facet; source "$(dirname "$0")/lib.sh"
CFG=configs/facet_qwen14b.yaml; CFG32=configs/facet_qwen32b.yaml; DATA=rs-facet-qwen14b-v1; D=work/datasets/$DATA
REPO=work/sources/facet-rs__facet; E=work/eval/facet; mkdir -p $E/style results/facet
STAGE=${1:-all}  # "all" runs every stage except leakage
run() { case "$STAGE" in all) [ "$1" != leakage ] ;; *) [ "$STAGE" = "$1" ] ;; esac; }

if run testset; then  # real added blocks after the cutoff -> keep those cargo build+test can judge
  [ -s $E/test_candidates_raw.jsonl ] || $PY -m data.test_set $REPO 2026-01-01T00:00:00+00:00 $E/test_candidates_raw.jsonl rust >> work/logs/facet.log
  [ -s $E/test_candidates.jsonl ] || $PY -m data.test_set_rust $REPO $E/test_candidates_raw.jsonl $E/test_candidates.jsonl >> work/logs/facet.log
  $PY -m eval.exec_rust $REPO $E/test_candidates.jsonl $E/exec.jsonl 4 >> work/logs/facet_exec.log 2>&1 || die "exec check"
  $PY -m data.eval_set $E/test_candidates.jsonl $E/exec.jsonl $E/eval.jsonl | tee -a work/logs/facet.log
  cp $E/test_candidates.summary.json results/facet/test_set.json
fi
if run data; then  # paired AST + random holes from the cutoff snapshot, 5% of files held out
  [ -d $D ] || $PY -m data.build_repo $CFG work/sources/facet-cutoff d3cd33f75da7b80fc0134ec7346c26512d3ada72 $DATA facet-rs/facet rust 10000 \
    2>&1 | grep -v "longer than" >> work/logs/facet.log
  [ -f $D/validation_subset.json ] || $PY -m data.val_subset --version $DATA --size 128 >> work/logs/facet.log
fi
if run train; then
  for arm in ast random; do train_arm $CFG $DATA $arm facet-qwen14b-$arm; done
fi
if run eval; then
  declare -A AD=([qwen14b-base]="" [facet-ast]="--adapter $(adapter facet-qwen14b-ast)" [facet-random]="--adapter $(adapter facet-qwen14b-random)" [qwen32b-base]="")
  [ -n "${AD[facet-ast]#--adapter }" ] && [ -n "${AD[facet-random]#--adapter }" ] || die "train both arms first"
  gen() { log "generate $2 on $(basename $3)"; $PY -m eval.repo_eval generate ${5:-$CFG} $2 $3 $4 ${AD[$1]} >> work/logs/facet_gen_$2.log 2>&1 || die "generate $2"; }
  score() { log "score $1"; $PY -m eval.repo_eval score $REPO $E/eval.jsonl $E/$1.completions.jsonl $E/$1.scores.jsonl 3 --lang rust >> work/logs/facet_score_$1.log 2>&1 || die "score $1"; }
  python3 - <<P
import json
ex = [json.loads(l) for l in open("$E/eval.jsonl")]
open("$E/eval_sensitive.jsonl", "w").write("".join(json.dumps(e) + "\n" for e in ex if e["sensitive"]))
open("$E/eval_rest.jsonl", "w").write("".join(json.dumps(e) + "\n" for e in ex if not e["sensitive"]))
P
  for m in qwen14b-base facet-ast facet-random; do gen $m $m $E/eval_sensitive.jsonl $E; gen $m $m $E/eval_rest.jsonl $E; done
  gen qwen32b-base qwen32b-base $E/eval_sensitive.jsonl $E $CFG32
  for m in qwen14b-base facet-ast facet-random qwen32b-base; do score $m; done                     # pass@1 + exact match
  for m in qwen14b-base facet-ast facet-random; do                                                   # char perplexity
    $PY -m eval.repo_eval ppl $CFG $m $E/eval.jsonl $E ${AD[$m]} >> work/logs/facet_ppl_$m.log 2>&1 || die "ppl $m"
  done
  for arm in ast random; do                                                                          # 'their style'
    [ -f $E/style/sample150_$arm.jsonl ] || python3 -c "
import random; ex = open('$D/validation_$arm.jsonl').readlines(); idx = sorted(random.Random(20261123).sample(range(len(ex)), 150))
open('$E/style/sample150_$arm.jsonl', 'w').write(''.join(ex[i] for i in idx))"
    for m in qwen14b-base facet-ast facet-random; do gen $m $m-val$arm $E/style/sample150_$arm.jsonl $E/style; done
  done
  $PY -m eval.summarize_facet $E results/facet/summary.json --extra qwen32b-base | tee -a work/logs/facet.log
  $PY -m eval.style_check_facet $E/style $D results/facet/style_check.json >> work/logs/facet.log
  $PY -m eval.hole_sizes $D $E/eval.jsonl results/facet/hole_sizes.json >> work/logs/facet.log
  $PY -m eval.compare_cards $E $E/eval.jsonl results/facet/summary.json $E/compare_cards.html >> work/logs/facet.log  # shows code: stays in work/
  log "facet: results/facet/summary.json written"
fi
