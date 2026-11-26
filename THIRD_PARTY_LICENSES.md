# Third-party code, data and models

This repository contains only our own code, configs, result summaries and run logs (Apache-2.0, see `LICENSE`).
It does **not** redistribute third-party source code, benchmark text, training data or model weights. Everything
below is fetched by `scripts/fetch.sh` at a pinned commit or revision. Each item keeps its own licence.

Licences were checked on 2026-11-25 against the licence files in the pinned checkouts. The Qwen licence was
checked against the Hugging Face model pages.

## Repositories used for training and evaluation

| Used for | Repository | Pinned commit(s) | Licence |
|---|---|---|---|
| Rust FIM experiment (train / validation / test) | [paradigmxyz/reth](https://github.com/paradigmxyz/reth) | `84caec584e79ddef33572394f787d2d1203a23c7` | MIT OR Apache-2.0 |
| Repo-specific, Rust | [facet-rs/facet](https://github.com/facet-rs/facet) | cutoff snapshot `d3cd33f75da7b80fc0134ec7346c26512d3ada72`, history up to `65bae5c31a7ce401bc44630fb96250ea884cfd3e` | MIT OR Apache-2.0 |
| Repo-specific, Go | [telekom/controlplane](https://github.com/telekom/controlplane) | cutoff snapshot `9164dcafe005913bf2c6e4870db776548c66662f`, history up to `cf866975bcaec72810622aa1f8869ead03c6fced` | Apache-2.0 for code (REUSE: CC0-1.0 / CC-BY-4.0 for some non-code files) |
| Multi-language experiment (Python, TypeScript, Go, C++) | 12 training-pool + 10 held-out repos per language, listed with commits in `configs/repos/<lang>_pins.json` | see pins | MIT, Apache-2.0 or BSD only (selection rule recorded in `configs/repos/rust_heldout_repos.json` and `results/multilang/summary.md`) |
| Rust held-out set | 15 repos created after 2024, listed with commits in `configs/repos/rust_heldout_repos.json` | see file | Apache-2.0 (8 repos), MIT (7 repos) |

Report files under `results/multilang/leakage/` name repository paths and contain similarity numbers. Quoted code lines have been
removed from them.

## Benchmark

| Dataset | Revision | Licence |
|---|---|---|
| [gonglinyuan/real_fim_eval](https://huggingface.co/datasets/gonglinyuan/real_fim_eval) (Real-FIM-Eval) | `a36062544c7ed6c4e5ffb8dad8536fc7777e1f36` | CC-BY-4.0 |

We publish example ids and scores only, never benchmark text.

## Models

| Model | Revision | Licence | How it is used here |
|---|---|---|---|
| [Qwen/Qwen2.5-Coder-14B](https://huggingface.co/Qwen/Qwen2.5-Coder-14B) | `f2ad5164aade432d6d56c24bb71589184d5d613d` | Apache-2.0 | base for the repo-specific LoRA adapters |
| [Qwen/Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B) | `2e12b5f7bc878d424d222e224ed40aee564ec45f` | Apache-2.0 | evaluated as a baseline only |
| [deepseek-ai/deepseek-coder-6.7b-base](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base) | `ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912` | DeepSeek License Agreement v1.0 (use-based restrictions) | base for the multi-language AST vs random adapters (not released) |
| [deepseek-ai/deepseek-coder-1.3b-base](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) | `c919139c3a9b4070729c8b2cca4847ab29ca8d94` | DeepSeek License Agreement v1.0 | 1.3B sanity check only |

The facet LoRA adapters (see `model_cards/`) are derivative works of Qwen2.5-Coder-14B (Apache-2.0). They were
trained on facet code (MIT OR Apache-2.0) and are released under Apache-2.0 with attribution to both.
