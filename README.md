# fim-coder-model

**Train your own Copilot on your own repo, and prove it got better.** Fine-tune an open code model for
fill-in-the-middle (FIM) autocomplete on one repository's history, then evaluate it on code the team wrote *after* the
training snapshot, scored by whether the project still **builds and passes its tests**.

**[Results](#results) · [Reproduce](#quickstart-reproduce-the-main-result) · [Model cards](model_cards/) · [Write-up](blog/blog.md)**

> Everything here ran on one NVIDIA DGX Spark. Code is Apache-2.0. Third-party repositories, benchmark data and
> model weights are fetched at pinned commits and are not redistributed (see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)).

## Highlights

- **+9.8 points pass@1 from a day of training.** Qwen2.5-Coder-14B + LoRA trained on facet-rs/facet reaches 65.1%
  pass@1 (cargo build + tests) on 1,056 real post-cutoff edits, vs 55.4% for the base model.
- **Matches a model twice its size.** It is level with base Qwen2.5-Coder-32B (63.5%; difference +1.6, 95% CI −0.9
  to +4.3) and takes about a third less time per completion.
- **The same direction in Go.** On telekom/controlplane it gets +4.3 points over base 14B (CI +1.8 to +6.8) and
  matches the 32B.
- **AST-aware holes don't help.** Holes cut along syntax-tree nodes never beat random holes of the same length:
  across Rust, Python, TypeScript, Go and C++ (perplexity) and on facet (pass@1). Most of the AST adapter's gain was
  learning when to stop.

## Overview

| | |
|---|---|
| Task | single-file fill-in-the-middle (prefix + suffix → middle), greedy decoding, ≤ 256 new tokens |
| Base model | Qwen2.5-Coder-14B (repo-specific runs); DeepSeek-Coder-6.7B (AST vs random across languages) |
| Fine-tuning | LoRA r=16, α=32, all attention + MLP projections; lr 1e-4 cosine, 1 epoch, batch 16, seed 42, bf16, no quantization |
| Training data | the repo at a cutoff commit only; paired AST / random holes (same file, same length); 5% of files held out |
| Test data | every 1–30-line added block in post-cutoff commits (≤ 5 per commit), kept only if the real code builds + passes tests and deleting it breaks them |
| Metrics | pass@1 (build + tests), exact match, character perplexity; paired bootstrap 95% CIs over commits, B = 2000 |
| Hardware | 1× NVIDIA DGX Spark (GB10, 128 GB unified memory); 14B LoRA peaks at 37.5 GB, ~25 h per facet run |

## Results

**facet-rs/facet (Rust)**: cutoff 2026-01-01; 1,056 build+test-checkable edits from 391 commits.

| model | pass@1 | exact match | char perplexity ↓ |
|---|---|---|---|
| Qwen2.5-Coder-14B (base) | 55.4% | 35.1% | 1.1378 |
| **14B + LoRA, random holes** | **65.1%** | **41.9%** | **1.1360** |
| 14B + LoRA, AST holes | 63.2% | 40.8% | 1.1433 |
| Qwen2.5-Coder-32B (base) | 63.5% | 39.7% | – |

| comparison (pass@1) | difference | 95% CI | |
|---|---|---|---|
| random − base 14B | +9.8 | [+7.2, +12.3] | beats |
| random − base 32B | +1.6 | [−0.9, +4.3] | matches |
| AST − random | −2.0 | [−4.1, +0.2] | no better |
| random − base 14B, examples where no model hit the token limit | +3.8 | [+1.4, +6.2] | beats |

**telekom/controlplane (Go)**: cutoff 2026-05-01; 632 build+test-checkable edits from 193 commits.

| model | pass@1 | exact match |
|---|---|---|
| Qwen2.5-Coder-14B (base) | 54.4% | 34.2% |
| **14B + LoRA, random holes** | **58.7%** | **36.6%** |
| DeepSeek-Coder-6.7B (base) | 55.5% | 33.0% |
| Qwen2.5-Coder-32B (base) | 56.2% | 36.1% |

LoRA − base 14B: +4.3 [+1.8, +6.8]. LoRA − 32B: +2.5 [0.0, +5.3], which is a match.

**AST vs random holes**, DeepSeek-Coder-6.7B, Real-FIM-Eval Add, character perplexity. A positive AST − random means
random holes are better.

| language | n | base | AST | random | AST − random [95% CI over repos] |
|---|---|---|---|---|---|
| Rust | 2,747 | 1.1816 | 1.1534 | **1.1507** | +0.0027 [+0.0024, +0.0031] |
| Python | 3,687 | 1.1749 | 1.1745 | 1.1744 | +0.0001 [−0.0005, +0.0004] |
| TypeScript | 1,671 | 1.1794 | 1.1784 | **1.1767** | +0.0017 [+0.0016, +0.0020] |
| Go | 1,312 | 1.1742 | 1.1749 | **1.1727** | +0.0022 [+0.0020, +0.0024] |
| C++ | 1,643 | 1.1538 | 1.1557 | **1.1542** | +0.0016 [+0.0012, +0.0020] |

Full numbers with every split (source vs test files, human vs AI-tagged commits, stop effect, held-out repos) are in
`results/`: `facet/summary.json`, `controlplane/summary.json` and `multilang/summary.md`.

## Quickstart: reproduce the main result

```bash
bash scripts/setup.sh           # .venv with pinned requirements + unit tests
bash scripts/fetch.sh facet     # facet-rs/facet at pinned commits, Qwen2.5-Coder-14B and -32B weights
bash scripts/facet.sh           # test set -> training data -> train AST + random -> evaluate -> results/facet/summary.json
```

`facet.sh` runs in stages (`testset`, `data`, `train`, `eval`), and every stage is resumable: rerun the same command
after an interruption. Rough timings on one DGX Spark:

| stage | time |
|---|---|
| build+test check of the candidate edits | hours (CPU) |
| training data | 3 minutes |
| training | ~25 h per arm |
| evaluation, including the 32B | about a day |

It needs `rustup`; facet pins its own toolchain. GPU memory: ~40 GB for 14B training, ~70 GB for 32B inference.

## Other experiments

```bash
bash scripts/fetch.sh controlplane && bash scripts/fetch.sh toolchains && bash scripts/controlplane.sh   # Go repo
bash scripts/fetch.sh rust && bash scripts/rust.sh                                                       # AST vs random, Rust
bash scripts/fetch.sh multilang && bash scripts/multilang.sh go                                          # ... python | typescript | go | cpp
```

`rust.sh leakage` and `multilang.sh <lang> leakage` rerun the leakage checks against the benchmark and held-out
repos, plus the READY/STOP gate. The published reports are in `results/multilang/leakage/`.

## Repository layout

```
core/        shared: config + pinned sources (config.py), FIM prompt formats (fim_format.py), model loading, languages
data/        test_set.py      post-cutoff test candidates from git history (leakage gate on commit dates)
             test_set_rust.py Rust: crates that existed at the cutoff only, at most 5 examples per commit
             eval_set.py      keep candidates that build+test can judge ("sensitive" = deleting the hole breaks them)
             build_repo.py    repo-specific training pairs (paired AST + random holes) from the cutoff snapshot
             build_reth.py, build_multilang.py, build_heldout.py   AST vs random datasets; holes.py, split.py
             leakage/         decontamination, MinHash near-duplicates, overlap reviews, READY/STOP gate
train/       train.py         LoRA trainer (loss on middle + end token only), resumable; optional private W&B logging
eval/        repo_eval.py     generate completions, char perplexity, and build+test scoring (exec_rust.py / exec_go.py)
             perplexity.py    teacher-forced character perplexity (Real-FIM-Eval, held-out sets)
             summarize_*.py, style_check_*.py, stop_effect_controlplane.py, hole_sizes.py, compare_cards.py
configs/     one YAML per experiment/model; repos/ (pinned repo lists, final exclude lists); leakage/ (check settings)
scripts/     setup.sh, fetch.sh, facet.sh, controlplane.sh, rust.sh, multilang.sh (lib.sh = shared helpers)
results/     every published summary: facet/, controlplane/, multilang/ (+ leakage/ reports)
runs/        logs behind the results: training (config, loss curve, manifest), eval (per-example scores), pipelines
model_cards/ Hugging Face cards for the two facet LoRA adapters (generated from results/ by make_cards.py)
blog/        write-up + charts (make_charts.py)
tests/       pytest: holes, FIM formats, label masks, scoring, Rust regression
work/        (gitignored) everything fetched or generated: sources, models, datasets, runs, eval outputs, caches
```

All Python entry points run as modules from the repo root, for example `python -m eval.summarize_facet …`. Each
file's docstring gives its exact usage.

## Configs

Each YAML pins everything a run depends on: model (local path + revision), FIM `format`, `data` (seed, hole density
`bytes_per_example`, `max_seq_tokens`), `lora`, `training`, `evaluation` (context window) and `wandb`. W&B is off by
default; when enabled it logs configs and scalar metrics only, to a project that must already be private. The
`sources` block pins reth and Real-FIM-Eval and is shared by every config.

## Run logs

Under `runs/` there are three kinds of log:
- `runs/training/<run>/`: `config.yaml`, `metrics.jsonl` (loss per update plus validation loss every 50 updates),
  `console.log` and `manifest.json` (package versions, data hashes, wall time, peak memory). Local paths and W&B
  details are removed.
- `runs/eval/`: per-example perplexity records (ids and numbers only).
- `runs/pipeline_logs/`: orchestration logs.

Checkpoints, completions and anything containing third-party code are not published.

## Reproducibility checks

Before publishing, the CPU-side pipeline was re-run from this layout and compared with the files behind the published
results. All of these come out **byte-identical**:
- **facet test set:** raw candidates from git history (39,251), the Rust filter (2,096 candidates from 552 commits),
  and the eval set (1,817 usable, 1,089 sensitive).
- **facet training data:** all four pair files, the file split and the validation subset.
- **Multi-language data (spot-checked on Go and Python):** the Go training and validation pairs, and the locked
  held-out sets for Go and Python, using the same two-step build as the original runs (`scripts/multilang.sh`).
- **Summaries:** every summary in `results/facet/` and `results/controlplane/`, regenerated from the original
  completions and scores, plus the Rust, Go and C++ multi-language summaries regenerated from `runs/eval/`. Python and
  TypeScript match in every number; their files lack one field the summarizer gained later (`skipped_over_context`).
- **Model cards and charts:** regenerated from `results/`.

Not re-run for the release:
- the cargo / go build+test execution check (hours of CPU);
- GPU steps (training, generation, perplexity).

The published results are their original outputs.

## Limitations

- **Small sample.** One seed per arm and one LoRA recipe, with no hyperparameter tuning for either arm. Two repos
  for the repo-specific result.
- **In-file context only.** Completions see only the current file, never other files of the repo.
- **Mostly stopping.** Most of the gain is the model learning where a completion ends. With limit hits excluded,
  facet random still gains +3.8 points; the AST adapter and the controlplane adapter do not.
- **Wide CIs on small subsets.** AI-tagged and bot-authored commits are small subsets (29 and 14 facet examples).
- **Inferred Edit layout.** The Real-FIM-Eval Edit layout is inferred from the dataset card, so Edit results are
  provisional. The Add results are primary.

## License and credits

Apache-2.0 (`LICENSE`, `NOTICE`). Inspired by [Build your own GitHub Copilot](https://prvn.sh/build-your-own-github-copilot).
Credits:
- facet-rs/facet, telekom/controlplane and paradigmxyz/reth, whose public histories made the time split possible;
- [Real-FIM-Eval](https://huggingface.co/datasets/gonglinyuan/real_fim_eval) (CC-BY-4.0);
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-14B) (Apache-2.0);
- [DeepSeek-Coder](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base).
