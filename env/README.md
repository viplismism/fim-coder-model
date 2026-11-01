# Environment

One project-local virtual environment, `.venv/`, shared by both experiment arms. The system Python
(3.12.3, Ubuntu 24.04) and the GPU driver were not changed. `requirements.lock` is the full `pip freeze`.

## How it was built (2026-11-01, NVIDIA GB10 / aarch64, driver 580.178.04, CUDA 13.0)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# torch: PyTorch's CUDA 13.0 index (aarch64 wheel)
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1"
# everything else: PyPI
.venv/bin/python -m pip install "transformers==5.22.1" "tokenizers==0.23.3" "huggingface-hub==2.4.0" \
    "datasets==5.1.0" "accelerate==1.15.0" "peft==0.21.2" "trl==1.17.1" "bitsandbytes==0.50.2" \
    "datatrove==0.10.1" "tree-sitter==0.26.0" "tree-sitter-rust==0.24.2" pyyaml
.venv/bin/python -m pip install "xxhash==3.8.2" orjson "pytest==9.1.1"
```

To recreate exactly: `pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1`, then
`pip install -r env/requirements.lock` (PyPI).

## Compatibility notes found while resolving

- `transformers==4.49.0` (the old pin) cannot coexist with `datatrove`: it needs `huggingface_hub<1.0`,
  datatrove 0.10.1 needs `>=1.6`. The current stack was tested instead (see `checks/env/`).
- `datatrove` 0.10.1 + `xxhash` 4.x fails (`TypeError: Strings must be encoded before hashing`).
  datatrove declares `xxhash<4` only in its `processing` extra, so pip picked 4.0.2; pinned to 3.8.2.
- datatrove's `JsonlReader` needs `orjson`.
- `pip check` reports `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform` (a wheel tag
  complaint); `torch.backends.cusparselt.is_available()` is `True`.
- `trl` is installed as requested but not used: the trainer uses `transformers.Trainer` with explicit labels.

## Verified (checks/env/report.json, checks/env/tokenizer_roundtrip.py)

- CUDA execution on GB10 (compute capability 12.1; torch ships sm_120 code, which runs on 12.1).
- bitsandbytes loads `libbitsandbytes_cuda130.so` (has native sm_121); NF4 4-bit matmul works.
- 4-bit NF4 DeepSeek-Coder-6.7B base loads (3.65 GB), LoRA forward + backward give finite gradients
  (no optimizer step was taken). A fresh adapter reproduces the base exactly without
  `prepare_model_for_kbit_training`; with it, logits differ (max 2.80) because it upcasts norms/embeddings.
- Tokenizer: exact encode/decode round trip on all 1,326 reth files and all Rust benchmark fields,
  except 8 benchmark examples containing characters that DeepSeek's own `tokenizer.json` maps to
  13 byte-fragment added tokens (ids 32000-32012) which decode to U+FFFD. Same ids as the raw
  `tokenizers` backend, so not a transformers issue; scoring never decodes, so scores are unaffected.
- transformers 5.x does not add BOS by default for this tokenizer: the pipeline adds it explicitly.
