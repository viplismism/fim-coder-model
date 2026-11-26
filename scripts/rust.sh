#!/usr/bin/env bash
# AST vs random FIM holes on Rust (paradigmxyz/reth, package split), DeepSeek-Coder-6.7B + LoRA, scored by character
# perplexity on the reth test split, Real-FIM-Eval Rust (Add = primary) and a locked 15-repo held-out set.
#   bash scripts/rust.sh [leakage|data|train|eval|all]   (default: data train eval; leakage reruns the overlap checks)
# Needs: scripts/setup.sh and `scripts/fetch.sh rust`.
LOG_NAME=rust; source "$(dirname "$0")/lib.sh"
CFG=configs/rust_ast_vs_random.yaml; DATA=rust-fim-v2; HELD=heldout-v1; E=work/eval/rust; mkdir -p $E work/leakage
STAGE=${1:-all}  # "all" runs every stage except leakage
run() { case "$STAGE" in all) [ "$1" != leakage ] ;; *) [ "$STAGE" = "$1" ] ;; esac; }

if run leakage; then  # reth vs benchmark exact/mention overlap, near-duplicates across splits, held-out repos vs everything
  $PY -m data.leakage.reth_benchmark_overlap --reth work/sources/reth --bench work/sources/real_fim_eval --out_dir work/leakage/reth_benchmark
  $PY -m data.leakage.neardup --split_json work/datasets/$DATA/split.json --out_dir work/leakage/neardup
  $PY -m data.leakage.heldout_overlap configs/repos/rust_heldout_repos.json work/leakage/heldout
  log "leakage reports in work/leakage/ (published copies: results/multilang/leakage/rust_*)"
fi
if run data; then
  [ -d work/datasets/$DATA ] || $PY -m data.build_reth --version $DATA >> work/logs/rust.log 2>&1 || die "build $DATA"
  [ -f work/datasets/$DATA/validation_subset.json ] || $PY -m data.val_subset --version $DATA --size 128 >> work/logs/rust.log
  [ -d work/datasets/$HELD ] || $PY -m data.build_heldout --version $HELD >> work/logs/rust.log 2>&1 || die "build $HELD"
fi
if run train; then for arm in ast random; do train_arm $CFG $DATA $arm $arm; done; fi
if run eval; then  # one-time final scoring: test split + benchmark, then the locked held-out set
  for name in base ast random; do
    ad=""; [ $name != base ] && ad="--adapter $(adapter $name)"
    [ -f $E/${name}_final/summary.json ] || $PY -m eval.perplexity --config $CFG --tasks reth_ast,reth_random,reth_l2r,bench_add,bench_edit \
      --data_version $DATA --split test --final $ad --out $E/${name}_final >> work/logs/rust_eval.log 2>&1 || die "final $name"
    [ -f $E/${name}_heldout/summary.json ] || $PY -m eval.perplexity --config $CFG --tasks heldout_ast,heldout_random,heldout_l2r \
      --final_heldout --heldout_version $HELD $ad --out $E/${name}_heldout >> work/logs/rust_eval.log 2>&1 || die "heldout $name"
  done
  $PY -m eval.summarize_rust final $E/base_final $E/ast_final $E/random_final results/multilang/rust_final.json | tee -a work/logs/rust.log
  $PY -m eval.summarize_rust heldout $E/base_heldout $E/ast_heldout $E/random_heldout results/multilang/rust_heldout.json | tee -a work/logs/rust.log
fi
