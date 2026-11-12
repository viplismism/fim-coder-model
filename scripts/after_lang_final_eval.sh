#!/usr/bin/env bash
# Wait for a language's final-eval pipeline to finish, summarise it, commit + push ONLY the summary (no data/logs).
# Usage: bash scripts/after_lang_final_eval.sh <lang> <pipeline_log>
set -uo pipefail
cd "$(dirname "$0")/.."
LANG_=$1; PIPE=$2; OUT=results/${LANG_}_final_eval_summary.json; LOG=logs/after_${LANG_}_final_eval.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }
log "waiting for $PIPE"
until grep -qE "final evaluation finished|FAILED" "$PIPE"; do sleep 120; done
if grep -q FAILED "$PIPE"; then log "pipeline FAILED; not summarising"; tail -3 "$PIPE" >> "$LOG"; exit 1; fi
dir() { grep "done: logs/.*_${LANG_}-$1_eval_final" "$PIPE" | tail -1 | sed 's/.*done: //'; }
B=$(dir base); A=$(dir ast); R=$(dir random)
log "dirs: base=$B ast=$A random=$R"
.venv/bin/python checks/multilang/summarize_final.py "$LANG_" "$B" "$A" "$R" "$OUT" >> "$LOG" 2>&1 || { log "summary FAILED"; exit 1; }
log "summary written: $OUT"
git add "$OUT" checks/multilang/summarize_final.py scripts/after_lang_final_eval.sh
BAD=$(git diff --cached --name-only | grep -E '^(data|logs|sources|scratch|\.cache|\.venv)/' || true)
SECRET=$(git diff --cached | grep -E 'wandb_v1_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|WANDB_API_KEY=[A-Za-z0-9]' || true)
if [ -n "$BAD$SECRET" ]; then log "safety check failed; not committing"; git reset -q; exit 1; fi
git commit -q -m "Python final evaluation summary (Real-FIM-Eval Add/Edit, held-out) with paired repo-level CIs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "committed $(git rev-parse --short HEAD)"
git push origin fresh-start >> "$LOG" 2>&1 && log "pushed fresh-start" || log "push FAILED"
log "all done"
