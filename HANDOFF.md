# Training handoff — run the FIM training on your GPU box

Thanks for running this! It's fully automated — one command does everything:
clone → install → download data → train → benchmark → push results back.

## Requirements
- A GPU with **24 GB+ VRAM** (RTX 4090 works; 80 GB A100/H100 is faster and lets the
  benchmark run without tweaks)
- **~50 GB free disk**
- NVIDIA driver new enough for **CUDA ≥ 12.4** (check with `nvidia-smi`)

## Run it (inside tmux so it survives disconnects)

```bash
tmux new -s fim

# secrets (paste the values vipul sent you):
export WANDB_API_KEY=<value>     # streams the live dashboard to vipul's W&B
export HF_TOKEN=<value>          # auto-pushes the trained model back to vipul
export NTFY_TOPIC=<value>        # phone ping when done
export PUSH_TO_HF=1              # turn on auto-push

# launch:
curl -sL https://raw.githubusercontent.com/viplismism/fim-coder-model/main/scripts/run_full_training.sh | bash 2>&1 | tee run.log

# detach and walk away:  press Ctrl+b , then d
```

Reattach anytime to check on it: `tmux attach -t fim`

## What it does (unattended, ~8–22 h depending on GPU)
1. Clones the repo + installs the pinned, tested dependencies
2. Downloads the dataset from Hugging Face (no extra setup)
3. Trains the LoRA adapter (1 epoch over the full dataset)
4. Benchmarks the new model vs the base model
5. Pings the phone topic with the result numbers
6. Pushes the adapter + results to a **private** HF repo for vipul

## When it's done
It prints a `=== ALL DONE ===` block with the result summary. Nothing else needed —
everything is already uploaded. You can `exit` the box.

## Knobs (optional, set before launching)
- `MAX_TRAIN_SAMPLES=20000` — train on a subset instead of all ~47k
- `EPOCHS=1` — number of passes (1 is the default and recommended)
