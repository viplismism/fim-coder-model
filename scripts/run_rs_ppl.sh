#!/usr/bin/env bash
# Character perplexity (teacher-forced, real added code) for Qwen-14B base and fine-tuned on the 891 common examples.
# Starts as soon as the fine-tuned full generation has finished (the pipeline then only waits for the 32B download).
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PL=$(ls -t logs/*_repo_specific_pipeline.log | head -1); R=checks/repo_specific; EX=$R/controlplane_eval_common.jsonl; OUT=results/repo_specific/full
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PL"; }
until [ "$(grep -c 'generated qwen14b-ft' $PL)" -ge 2 ] || grep -q "generation FAILED: qwen14b-ft" $PL; do sleep 60; done
AD=$(ls -dt logs/*_qwen14b-random_train_s42 | head -1)/checkpoints/final_adapter
log "char perplexity: qwen14b-base"
.venv/bin/python $R/gen_eval.py ppl configs/rs_qwen14b.yaml qwen14b-base $EX $OUT > logs/rs_ppl_qwen14b-base.log 2>&1 && log "ppl done: qwen14b-base" || log "ppl FAILED: qwen14b-base"
log "char perplexity: qwen14b-ft"
.venv/bin/python $R/gen_eval.py ppl configs/rs_qwen14b.yaml qwen14b-ft $EX $OUT --adapter $AD > logs/rs_ppl_qwen14b-ft.log 2>&1 && log "ppl done: qwen14b-ft" || log "ppl FAILED: qwen14b-ft"
