#!/usr/bin/env bash
# Resume step 0 after a shutdown: restarts the memory watchdog, the three validation generation processes
# (they skip items already saved in results/val_structural/gen/*.jsonl), the results web server, and the chain
# that runs compile checks + structural report (CPU) and the 1.3B sanity check (GPU) once generation is complete.
# Usage: bash scripts/resume_step0.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
if pgrep -f "fimexp\.(gen_batch|train|score)" >/dev/null; then echo "experiment processes already running"; exit 1; fi
tmux new-session -d -s fim-watch "bash checks/memwatch.sh results/val_structural/memory.tsv 16 'fimexp.gen_batch|fimexp.train|fimexp.score|cargo check|rustc'"
for m in base ast random; do
  tmux new-session -d -s gen-$m "HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv/bin/python -m fimexp.gen_batch --data_version rust-fim-v2 \
    --out results/val_structural/gen --max_answer_tokens 512 --batch_size 1 --merge_adapter --models $m >> results/val_structural/gen_$m.log 2>&1"
done
TS_IP=$(tailscale ip -4 2>/dev/null || true)
[ -n "$TS_IP" ] && tmux new-session -d -s fim-web "cd results && python3 -m http.server 8765 --bind $TS_IP"
sleep 5
tmux new-session -d -s fim-chain "bash scripts/run_step0_and_sanity.sh"
tmux ls
for m in base ast random; do echo "$m: $(wc -l < results/val_structural/gen/$m.jsonl)/$(wc -l < results/val_structural/gen/items.jsonl) generated"; done
