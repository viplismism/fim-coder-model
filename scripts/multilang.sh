#!/usr/bin/env bash
# AST vs random FIM holes for one language (python | typescript | go | cpp): 12 training-pool repos (10 train, 2
# validation) + 10 held-out repos, DeepSeek-Coder-6.7B + LoRA, scored by character perplexity on Real-FIM-Eval
# (Add = primary) and the held-out repos.
#   bash scripts/multilang.sh <lang> [leakage|data|train|eval|all]   (default: data train eval)
# data rebuilds the datasets from the committed inputs (configs/repos/<lang>_{pins,exclude_files,exclude_files_v1}.json);
# leakage reruns decontamination, overlap checks and the READY/STOP gate (reports -> results/multilang/leakage/).
# Needs: scripts/setup.sh and `scripts/fetch.sh multilang`.
L=${1:?usage: multilang.sh <python|typescript|go|cpp> [stage]}; LOG_NAME=multilang_$L; source "$(dirname "$0")/lib.sh"
declare -A VERSION=([python]=python-fim-v3 [typescript]=typescript-fim-v2 [go]=go-fim-v2 [cpp]=cpp-fim-v2)
V=${VERSION[$L]:?unknown language $L}; HELD=$L-fim-v1-heldout; CFG=configs/multilang_$L.yaml; E=work/eval/$L; mkdir -p $E
STAGE=${2:-all}  # "all" runs every stage except leakage
run() { case "$STAGE" in all) [ "$1" != leakage ] ;; *) [ "$STAGE" = "$1" ] ;; esac; }

if run data; then
  # Same two steps as the published runs: the first build (first-round decontamination list; none for Python, whose
  # first build predates decontamination) creates the locked held-out set; the final build reuses it unchanged and
  # uses the final exclude list (adds the train-vs-validation removals). Held-out sampling depends on that order.
  if [ ! -d work/datasets/$HELD ]; then
    X=""; [ -f configs/repos/${L}_exclude_files_v1.json ] && X="--exclude_files configs/repos/${L}_exclude_files_v1.json"
    $PY -m data.build_multilang --lang $L --pins configs/repos/${L}_pins.json --version $L-fim-v1 $X 2>&1 | grep -v "longer than" >> work/logs/multilang_$L.log
  fi
  [ -d work/datasets/$V ] || $PY -m data.build_multilang --lang $L --pins configs/repos/${L}_pins.json --version $V \
    --exclude_files configs/repos/${L}_exclude_files.json --reuse_heldout $HELD 2>&1 | grep -v "longer than" >> work/logs/multilang_$L.log
  [ -f work/datasets/$V/validation_subset.json ] || $PY -m data.val_subset --version $V --size 128 >> work/logs/multilang_$L.log
fi
if run leakage; then
  $PY -m data.leakage.decontaminate $L configs/repos/${L}_pins.json >> work/logs/multilang_$L.log     # pool vs benchmark + held-out
  $PY -m data.leakage.train_vs_validation $L $V >> work/logs/multilang_$L.log
  $PY -m data.leakage.decontaminate $L configs/repos/${L}_pins.json --pool_only $V >> work/logs/multilang_$L.log
  $PY -m data.leakage.overlap $L $V >> work/logs/multilang_$L.log
  $PY -m data.leakage.review_overlap $L $V >> work/logs/multilang_$L.log
  $PY -m data.leakage.gate $L $V $HELD | tee -a work/logs/multilang_$L.log
fi
if run train; then for arm in ast random; do train_arm $CFG $V $arm $L-$arm; done; fi
if run eval; then  # one-time final scoring
  for name in base ast random; do
    ad=""; [ $name != base ] && ad="--adapter $(adapter $L-$name)"
    [ -f $E/${name}_final/summary.json ] || $PY -m eval.perplexity --config $CFG --lang $L \
      --tasks bench_add,bench_edit,heldout_ast,heldout_random,heldout_l2r --final_heldout --heldout_version $HELD $ad \
      --out $E/${name}_final >> work/logs/multilang_${L}_eval.log 2>&1 || die "final $name"
  done
  $PY -m eval.summarize_multilang $L $E/base_final $E/ast_final $E/random_final results/multilang/${L}_final.json | tee -a work/logs/multilang_$L.log
fi
