#!/usr/bin/env bash
# telekom/controlplane (Go), cutoff 2026-05-01: Qwen2.5-Coder-14B + LoRA (random holes) vs base 14B, base 32B and
# base DeepSeek-Coder-6.7B, scored by go build + go test on real post-cutoff edits.
#   bash scripts/controlplane.sh [testset|data|train|eval|all]   (default: all; every stage is resumable)
# Needs: scripts/setup.sh, `scripts/fetch.sh controlplane`, `scripts/fetch.sh toolchains` (Go + envtest).
LOG_NAME=controlplane; source "$(dirname "$0")/lib.sh"
CFG=configs/controlplane_qwen14b.yaml; DATA=rs-controlplane-qwen14b-v1; D=work/datasets/$DATA
REPO=work/sources/telekom__controlplane; E=work/eval/controlplane; mkdir -p $E/style results/controlplane
STAGE=${1:-all}  # "all" runs every stage except leakage
run() { case "$STAGE" in all) [ "$1" != leakage ] ;; *) [ "$STAGE" = "$1" ] ;; esac; }

if run testset; then  # every model sees the same examples: those that fit DeepSeek-6.7B's 16K window
  [ -s $E/test_candidates.jsonl ] || $PY -m data.test_set $REPO 2026-05-01T00:00:00+00:00 $E/test_candidates.jsonl go >> work/logs/controlplane.log
  $PY -m eval.exec_go $REPO $E/test_candidates.jsonl $E/exec.jsonl 4 >> work/logs/controlplane_exec.log 2>&1 || die "exec check"
  $PY -m data.eval_set $E/test_candidates.jsonl $E/exec.jsonl $E/eval.jsonl --common_context configs/controlplane_deepseek67b.yaml | tee -a work/logs/controlplane.log
  cp $E/test_candidates.summary.json results/controlplane/test_set.json; cp $E/eval_dropped.json results/controlplane/eval_set_dropped_for_context.json
fi
if run data; then  # random holes from the cutoff snapshot (AST arm written alongside, not trained), 5% of files held out
  [ -d $D ] || $PY -m data.build_repo $CFG work/sources/controlplane-cutoff 9164dcafe005913bf2c6e4870db776548c66662f $DATA \
    2>&1 | grep -v "longer than" >> work/logs/controlplane.log
  [ -f $D/validation_subset.json ] || $PY -m data.val_subset --version $DATA --size 128 >> work/logs/controlplane.log
fi
if run train; then train_arm $CFG $DATA random qwen14b-random; fi
if run eval; then
  AD="--adapter $(adapter qwen14b-random)"; [ "$AD" != "--adapter " ] || die "train first"
  gen() { log "generate $2"; $PY -m eval.repo_eval generate $1 $2 $3 $4 ${5:-} >> work/logs/controlplane_gen_$2.log 2>&1 || die "generate $2"; }
  score() { log "score $1"; $PY -m eval.repo_eval score $REPO $E/eval.jsonl $E/$1.completions.jsonl $E/$1.scores.jsonl 4 >> work/logs/controlplane_score_$1.log 2>&1 || die "score $1"; }
  gen configs/controlplane_deepseek67b.yaml deepseek67b-base $E/eval.jsonl $E
  gen $CFG qwen14b-base $E/eval.jsonl $E
  gen $CFG qwen14b-ft $E/eval.jsonl $E "$AD"
  gen configs/controlplane_qwen32b.yaml qwen32b-base $E/eval.jsonl $E
  for m in deepseek67b-base qwen14b-base qwen14b-ft qwen32b-base; do score $m; done
  $PY -m eval.repo_eval ppl $CFG qwen14b-base $E/eval.jsonl $E >> work/logs/controlplane_ppl.log 2>&1 || die "ppl base"
  $PY -m eval.repo_eval ppl $CFG qwen14b-ft $E/eval.jsonl $E $AD >> work/logs/controlplane_ppl.log 2>&1 || die "ppl ft"
  for arm in random ast; do  # 'their style': the 5% held-out validation files
    gen $CFG qwen14b-base-val$arm $D/validation_$arm.jsonl $E/style
    gen $CFG qwen14b-ft-val$arm $D/validation_$arm.jsonl $E/style "$AD"
  done
  $PY -m eval.summarize_controlplane $E results/controlplane/summary.json qwen14b-ft qwen14b-base deepseek67b-base qwen32b-base | tee -a work/logs/controlplane.log
  $PY -m eval.stop_effect_controlplane $E results/controlplane/stop_effect.json >> work/logs/controlplane.log
  $PY -m eval.style_check_controlplane $E/style $D results/controlplane/style_check.json >> work/logs/controlplane.log
  log "controlplane: results/controlplane/summary.json written"
fi
