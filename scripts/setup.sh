#!/usr/bin/env bash
# Create .venv with the pinned environment and run the unit tests.
# Tested on NVIDIA DGX Spark (aarch64, CUDA 13.0, Python 3.12). For another CUDA version set TORCH_INDEX, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}" torch==2.14.1
.venv/bin/pip install -q -r requirements.txt
# the unit tests tokenize with the pinned DeepSeek-Coder tokenizer: fetch just its tokenizer files (a few MB) into work/hf-cache
HF_HUB_OFFLINE=0 .venv/bin/python -c "import core; from transformers import AutoTokenizer; \
  AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-6.7b-base', revision='ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912')" >/dev/null
.venv/bin/python -m pytest -q tests   # benchmark-dependent tests are skipped until `scripts/fetch.sh rust` has run
