#!/usr/bin/env bash
# Go final evaluation rerun (the first attempt stopped on an over-context Edit example; the scorer now skips and lists
# such benchmark examples). Waits for the language queue to finish so only one GPU job runs at a time.
cd "$(dirname "$0")/.."
until grep -q "queue finished" logs/20261113T112941+0530_lang_queue.log; do sleep 120; done
bash scripts/run_lang_final_eval.sh go configs/lang_go.yaml go-fim-v1-heldout
bash scripts/after_lang_final_eval.sh go $(ls -t logs/*_go_final_eval_pipeline.log | head -1)
