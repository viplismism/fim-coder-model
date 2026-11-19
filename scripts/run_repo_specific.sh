#!/usr/bin/env bash
# Overnight repo-specific pipeline (telekom/controlplane, cutoff 2026-05-01). One GPU job at a time; scoring (CPU)
# runs in the background, one scorer at a time (flock). Stages:
#  1. wait for the DeepSeek-6.7B base generation (already running) -> score
#  2. wait for Qwen-14B download -> build training data (95/5 file split) -> smoke-10 generation + score
#     -> full Qwen-14B base generation -> score
#  3. Qwen-14B LoRA: 5-update smoke run, then the full run (same recipe as before, random holes)
#     -> full generation with the adapter -> score
#  4. wait for Qwen-32B download -> smoke-10 + score -> full generation -> score
#  5. wait for all scoring -> summary -> commit + push the summary (no data, completions, logs or models)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=checks/repo_specific
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_SILENT=true
LOG=logs/$(date +%Y%m%dT%H%M%S%z)_repo_specific_pipeline.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }
EX=$R/controlplane_eval_common.jsonl; SMOKE=$R/controlplane_smoke10.jsonl
FULL=results/repo_specific/full; SM=results/repo_specific/smoke; mkdir -p $FULL $SM
REPO=sources_repo_specific/telekom__controlplane; DATA=rs-controlplane-qwen14b-v1
score_bg() {  # $1 dir, $2 name, $3 examples
  ( flock 9; $PY $R/gen_eval.py score $REPO $3 $1/$2.completions.jsonl $1/$2.scores.jsonl 4 > logs/rs_score_$2.log 2>&1
    echo "[$(date '+%F %T %Z')] scored $1/$2" >> "$LOG" ) 9>/tmp/rs_score.lock &
}
gen() {  # $1 config, $2 name, $3 examples, $4 out dir, [$5 adapter]
  log "generate $2 on $(wc -l < $3) examples"
  $PY $R/gen_eval.py generate $1 $2 $3 $4 ${5:+--adapter $5} > logs/rs_gen_$2.log 2>&1 && log "generated $2" || { log "generation FAILED: $2"; return 1; }
}
wait_model() { until [ -f models/$1/DOWNLOAD.json ]; do sleep 120; done; ! grep -q FAILED models/$1/DOWNLOAD.json; }

log "1. DeepSeek-6.7B base"
while tmux has-session -t rs-gen-deepseek 2>/dev/null; do sleep 60; done
[ $(wc -l < $FULL/deepseek67b-base.completions.jsonl) -ge $(wc -l < $EX) ] && log "generated deepseek67b-base" || log "WARNING: deepseek67b-base incomplete"
score_bg $FULL deepseek67b-base $EX

log "2. waiting for Qwen-14B download"
if wait_model Qwen2.5-Coder-14B; then
  log "Qwen-14B verified; building training data"
  if [ ! -d data/$DATA ]; then
    $PY $R/build_repo_data.py configs/rs_qwen14b.yaml sources_repo_specific/wt_cutoff 9164dcafe005913bf2c6e4870db776548c66662f $DATA 2>&1 | grep -v "longer than" >> "$LOG"
    $PY -m fimexp.val_subset --version $DATA --size 128 2>&1 | tail -3 >> "$LOG"
  fi
  gen configs/rs_qwen14b.yaml qwen14b-base $SMOKE $SM && $PY $R/gen_eval.py score $REPO $SMOKE $SM/qwen14b-base.completions.jsonl $SM/qwen14b-base.scores.jsonl 4 >> logs/rs_score_smoke.log 2>&1
  gen configs/rs_qwen14b.yaml qwen14b-base $EX $FULL && score_bg $FULL qwen14b-base $EX

  log "3. Qwen-14B LoRA: smoke (5 updates)"
  if $PY -m fimexp.train --config configs/rs_qwen14b.yaml --data_version $DATA --arm random --smoke_updates 5 > logs/rs_train_smoke.log 2>&1; then
    log "smoke ok: $(grep -m1 'peak' logs/rs_train_smoke.log | tail -c 200)"
    log "full training"
    if $PY -m fimexp.train --config configs/rs_qwen14b.yaml --data_version $DATA --arm random > logs/rs_train_full.log 2>&1; then
      AD=$(ls -dt logs/*_qwen14b-random_train_s42 | head -1)/checkpoints/final_adapter
      log "trained: $AD"
      gen configs/rs_qwen14b.yaml qwen14b-ft $SMOKE $SM $AD && $PY $R/gen_eval.py score $REPO $SMOKE $SM/qwen14b-ft.completions.jsonl $SM/qwen14b-ft.scores.jsonl 4 >> logs/rs_score_smoke.log 2>&1
      gen configs/rs_qwen14b.yaml qwen14b-ft $EX $FULL $AD && score_bg $FULL qwen14b-ft $EX
    else log "TRAINING FAILED (see logs/rs_train_full.log)"; fi
  else log "TRAINING SMOKE FAILED (see logs/rs_train_smoke.log); skipping training"; fi
else log "Qwen-14B download FAILED"; fi

log "4. waiting for Qwen-32B download"
if wait_model Qwen2.5-Coder-32B; then
  gen configs/rs_qwen32b.yaml qwen32b-base $SMOKE $SM && $PY $R/gen_eval.py score $REPO $SMOKE $SM/qwen32b-base.completions.jsonl $SM/qwen32b-base.scores.jsonl 4 >> logs/rs_score_smoke.log 2>&1
  gen configs/rs_qwen32b.yaml qwen32b-base $EX $FULL && score_bg $FULL qwen32b-base $EX
else log "Qwen-32B download FAILED"; fi

log "5. waiting for scoring"
wait
flock /tmp/rs_score.lock true
$PY $R/summarize_rs.py $FULL results/repo_specific/summary.json qwen14b-ft qwen14b-base deepseek67b-base qwen32b-base >> "$LOG" 2>&1
$PY $R/summarize_rs.py $SM results/repo_specific/smoke_summary.json qwen14b-ft qwen14b-base deepseek67b-base qwen32b-base >> "$LOG" 2>&1
git add results/repo_specific/summary.json results/repo_specific/smoke_summary.json $R/*.py $R/*.sh $R/PLAN.md $R/controlplane_dataset_stats.json \
    $R/controlplane_eval_common_dropped.json $R/build_test_cutoff.tsv $R/build_test_head.tsv configs/rs_*.yaml fimexp/psm.py scripts/run_repo_specific.sh .gitignore 2>>"$LOG"
BAD=$(git diff --cached --name-only | grep -E '^(data|logs|sources|sources_repo_specific|models|tools)/|\.jsonl$' || true)
SECRET=$(git diff --cached | grep -E 'hf_[A-Za-z0-9]{30,}|wandb_v1_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}' || true)
if [ -n "$BAD$SECRET" ]; then log "safety check failed; not committing"; git reset -q
else
  git commit -q -m "Repo-specific comparison (telekom/controlplane): pipeline, Qwen FIM format, build+test scorer, results summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin fresh-start >> "$LOG" 2>&1 && log "pushed" || log "commit/push failed"
fi
log "pipeline finished"
