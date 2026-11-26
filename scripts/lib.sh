# Shared helpers for the pipeline scripts (sourced, not run). Everything generated goes under work/ (gitignored).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-.venv/bin/python}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_SILENT=true
mkdir -p work/logs
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "work/logs/${LOG_NAME:-pipeline}.log"; }
die() { log "FAILED: $*"; exit 1; }

# latest finished adapter of a run, e.g. adapter facet-qwen14b-random  ->  work/runs/<ts>_facet-qwen14b-random_train_s42/...
adapter() { ls -d work/runs/*_"$1"_train_s42/checkpoints/final_adapter 2>/dev/null | sort | tail -1; }

# train one arm; skipped if finished, resumed from the latest checkpoint if interrupted (up to 3 times)
train_arm() {  # config data_version arm run_name
  local cfg=$1 data=$2 arm=$3 name=$4 last
  [ -n "$(adapter "$name")" ] && { log "train $name: already finished"; return 0; }
  for try in 0 1 2 3; do
    last=$(ls -d work/runs/*_"${name}"_train_s42 2>/dev/null | sort | tail -1)
    if [ -n "$last" ] && ls -d "$last"/checkpoints/checkpoint-* >/dev/null 2>&1; then
      log "train $name: resuming $last"; $PY -m train.train --config "$cfg" --data_version "$data" --arm "$arm" --resume_run_dir "$last" >> "work/logs/train_$name.log" 2>&1
    else
      log "train $name: starting"; $PY -m train.train --config "$cfg" --data_version "$data" --arm "$arm" >> "work/logs/train_$name.log" 2>&1
    fi
    [ -n "$(adapter "$name")" ] && { log "train $name: finished ($(adapter "$name"))"; return 0; }
  done
  die "train $name (see work/logs/train_$name.log)"
}
