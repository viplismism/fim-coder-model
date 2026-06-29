#!/usr/bin/env bash
#
# Self-contained FIM training run — hand this to anyone with a GPU box.
# Clones the repo, pulls the exact dataset from HF, trains (whitespace-safe
# stack + guard), benchmarks vs base, pings your phone, and (optionally)
# pushes the adapter to HF.
#
# USAGE (run inside tmux or with nohup so it survives disconnects):
#   export WANDB_API_KEY=...         # required for the live W&B dashboard
#   export HF_TOKEN=...              # required only if PUSH_TO_HF=1 (write token)
#   export NTFY_TOPIC=...            # optional: phone ping topic (ntfy.sh app)
#   bash run_full_training.sh 2>&1 | tee run.log
#
# Knobs (env vars, all optional):
#   MAX_TRAIN_SAMPLES=  (default: all 46,769; set e.g. 20000 for a smaller run)
#   EPOCHS=1
#   BENCH_SAMPLES=200   BENCH_K=1   (matches the 5k baseline for clean comparison)
#   PUSH_TO_HF=0        HF_REPO=viplismism/deepseek-coder-6.7b-fim-reth-v2
set -euo pipefail

# ---- config ----
REPO_URL="https://github.com/viplismism/fim-coder-model.git"
DATASET="viplismism/reth-fim-dataset"
BASE_MODEL="deepseek-ai/deepseek-coder-6.7b-base"
EPOCHS="${EPOCHS:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"        # empty = full dataset
BENCH_SAMPLES="${BENCH_SAMPLES:-200}"
BENCH_K="${BENCH_K:-1}"
PUSH_TO_HF="${PUSH_TO_HF:-0}"
HF_REPO="${HF_REPO:-viplismism/deepseek-coder-6.7b-fim-reth-v2}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
WORKDIR="${WORKDIR:-$HOME/fim-run}"

say() { echo -e "\n=== $* ===\n"; }
ping_phone() { [ -n "$NTFY_TOPIC" ] && curl -s -H "Title: $1" -d "$2" "ntfy.sh/$NTFY_TOPIC" >/dev/null || true; }

# ---- 0. sanity checks ----
say "ENV CHECK"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "no GPU"; exit 1; }
python3 -c "import sys; assert sys.version_info[:2]>=(3,10)" || { echo "need python>=3.10"; exit 1; }
[ -z "${WANDB_API_KEY:-}" ] && echo "WARN: WANDB_API_KEY not set -> W&B will log offline"
[ "$PUSH_TO_HF" = "1" ] && [ -z "${HF_TOKEN:-}" ] && { echo "PUSH_TO_HF=1 but HF_TOKEN not set"; exit 1; }

# ---- 1. clone repo ----
say "CLONE REPO"
rm -rf "$WORKDIR"; git clone --depth 1 "$REPO_URL" "$WORKDIR"; cd "$WORKDIR"

# ---- 2. venv + pinned deps (the whitespace-safe stack) ----
say "INSTALL DEPS"
python3 -m venv .venv && source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q "huggingface_hub[cli]"
python3 -c "import torch,transformers,trl,peft; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'| transformers',transformers.__version__,'trl',trl.__version__,'peft',peft.__version__)"

# ---- 3. pull the exact dataset (same split behind the 5k baseline) ----
say "DOWNLOAD DATASET"
mkdir -p data
hf download "$DATASET" --repo-type dataset reth_train.jsonl reth_test.jsonl --local-dir data
wc -l data/reth_train.jsonl data/reth_test.jsonl

# ---- 4. train (guard runs at startup; fails fast on a bad tokenizer stack) ----
say "TRAIN (epochs=$EPOCHS, max_train_samples=${MAX_TRAIN_SAMPLES:-all})"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TRAIN_ARGS=(--epochs "$EPOCHS")
[ -n "$MAX_TRAIN_SAMPLES" ] && TRAIN_ARGS+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
python training/train.py "${TRAIN_ARGS[@]}"

FINAL_MODEL=$(ls -dt training/runs/*/final_model | head -1)
echo "adapter saved at: $FINAL_MODEL"
ping_phone "FIM training done" "Training finished. Benchmarking next. Adapter: $FINAL_MODEL"

# ---- 5. benchmark vs base (same settings as the 5k baseline) ----
say "BENCHMARK (samples=$BENCH_SAMPLES, k=$BENCH_K)"
python evaluation/benchmark.py \
  --tuned "$FINAL_MODEL" --baseline "$BASE_MODEL" \
  --test_data data/reth_test.jsonl \
  --max_samples "$BENCH_SAMPLES" --k "$BENCH_K" \
  --output evaluation/results/full_bench.json

# ---- 6. summarize + ping ----
say "RESULTS"
SUMMARY=$(python3 - <<'PY'
import json,glob
d=json.load(open(sorted(glob.glob("evaluation/results/full_bench.json"))[-1]))
r=d["results"]; n=len(r); ag=lambda m,k: sum(x[m+"_metrics"][k] for x in r)
print(f"pass@1 tuned {ag('tuned','pass_at_1')}/{n} ({ag('tuned','pass_at_1')/n*100:.1f}%) vs base {ag('baseline','pass_at_1')}/{n} ({ag('baseline','pass_at_1')/n*100:.1f}%) | edit-sim {ag('tuned','edit_similarity')/n:.3f} vs {ag('baseline','edit_similarity')/n:.3f}")
PY
)
echo "$SUMMARY"
ping_phone "FIM benchmark done" "$SUMMARY"

# ---- 7. optional: push adapter + results to HF ----
if [ "$PUSH_TO_HF" = "1" ]; then
  say "PUSH TO HF: $HF_REPO (private)"
  hf upload "$HF_REPO" "$FINAL_MODEL" . --repo-type model --private \
    --commit-message "Full-data retrain adapter + eval ($SUMMARY)" --token "$HF_TOKEN"
  hf upload "$HF_REPO" evaluation/results/full_bench.json full_bench.json \
    --repo-type model --token "$HF_TOKEN"
  echo "pushed to https://huggingface.co/$HF_REPO (private — review before making public)"
fi

say "ALL DONE"
echo "$SUMMARY"
if [ "$PUSH_TO_HF" = "1" ]; then
  echo "✅ Model + results pushed to https://huggingface.co/$HF_REPO (private)."
  echo "   Nothing else to do — vipul has everything. You can exit the box."
else
  echo "⚠️  PUSH_TO_HF was off. Send vipul these 2 files:"
  echo "     - $WORKDIR/$FINAL_MODEL   (the trained adapter folder)"
  echo "     - $WORKDIR/evaluation/results/full_bench.json"
fi
