# AST-aware vs random hole selection for FIM fine-tuning: summary

Branch `fresh-start`, 2026-11-17. Per-language numbers: `results/<lang>_final_eval_summary.json` (Rust:
`results/final_eval_summary.json`, `results/heldout_eval_summary.json`).

## Question

When fine-tuning a code model for fill-in-the-middle (FIM), does choosing the training holes from the syntax tree
(one whole AST node per hole) beat random holes of the same length in the same files?

## Answer

**No.** AST-node holes did not beat matched random holes in any of the five languages on the primary score.
Random holes were significantly better in Rust, TypeScript, Go and C++, and tied in Python.

## Setup (identical for every language; only the data changes)

- **Model:** DeepSeek-Coder-6.7B base, bf16, LoRA r=16, alpha=32, dropout 0.05, all attention + MLP projections.
  1 epoch, batch 1 x 16 accumulation, lr 1e-4 cosine, warmup 5%, seed 42. A fresh adapter per arm from the same base.
- **Arms:** `ast`, a single tree-sitter node per hole (size-weighted; comments atomic; keyword/punctuation leaves
  excluded). `random`, the same hole length at a random position in the same file.
- **Format:** PSM FIM prompt (BOS, `<｜fim▁begin｜>` prefix `<｜fim▁hole｜>` suffix `<｜fim▁end｜>` middle). It
  contains code only; the language is never named. Loss is on the middle + EOS.
- **Data:**
  - **Rust:** one repo (reth), split by package.
  - **Python, TypeScript, Go, C++:** 12 repos each (10 train, 2 validation, seeded repo-level split) plus 10 fresh
    held-out repos. Selection rule: MIT/Apache/BSD licence, one repo per owner across all lists, no Real-FIM-Eval
    repo or name.
  - **Pair counts:** 7,549 train and 518 validation pairs per language, per-repo water-filled; 1,000 held-out pairs.
- **Decontamination:** file-level, written before use (`checks/multilang/DECONTAMINATION.md`). Training-pool files
  sharing a non-licence exact 5-line window or a verified MinHash near-duplicate (whole file, 30- or 10-line chunk,
  Jaccard >= 0.5 after licence-header removal) with the benchmark or held-out code were removed. Validation files
  copying training files were removed. Planted-copy controls: 20/20 whole-file copies caught in every new language;
  13-18/20 short fragments.
- **Metric:** teacher-forced character perplexity, exp(-sum answer logprob / answer chars), EOS excluded. It does
  not depend on the tokenizer.
- **Primary score:** Real-FIM-Eval **Add** for that language.
- **Decision rule, fixed in advance:** AST wins only if (AST - random) < 0 and the repo-level paired-bootstrap 95%
  CI upper bound < 0.

## Primary result: Real-FIM-Eval Add

| Language | n | base | AST | random | AST - random | repo-level 95% CI | AST better (repos) | gain vs base AST / random |
|---|---|---|---|---|---|---|---|---|
| Rust | 2,747 | 1.1816 | 1.1534 | 1.1507 | +0.0027 | [+0.0024, +0.0031] | 6/46 | 14.5% / 15.9% |
| Python | 3,687 | 1.1749 | 1.1745 | 1.1744 | +0.0001 | [-0.0005, +0.0004] | 22/71 | 0.2% / 0.3% |
| TypeScript | 1,671 | 1.1794 | 1.1784 | 1.1767 | +0.0017 | [+0.0016, +0.0020] | 1/28 | 0.5% / 1.4% |
| Go | 1,312 | 1.1742 | 1.1749 | 1.1727 | +0.0022 | [+0.0020, +0.0024] | 1/18 | -0.4% / 0.8% |
| C++ | 1,643 | 1.1538 | 1.1557 | 1.1542 | +0.0016 | [+0.0012, +0.0020] | 4/33 | -1.2% / -0.2% |

"Gain vs base" is the reduction in log character perplexity (uncertainty per character). A positive AST - random
means random is better.

## Secondary results

- **Edit (provisional; the prompt layout is inferred from the dataset figure).** AST - random is slightly negative
  in Python (-0.0007, CI crosses 0), TypeScript (-0.0006), Go (-0.0004) and C++ (-0.0004); the last three have
  repo-level CIs below 0. Rust is +0.0003 (CI crosses 0). The pattern is consistent but tiny, and it rests on an
  unconfirmed layout, so it does not change the verdict.
- **Held-out repos (scored once):**
  - Random-position holes and left-to-right scoring: random is better in every language (all CIs above 0).
  - AST-node holes: the two arms are within ±0.0004 everywhere. AST is marginally better in TypeScript and Go.
- **Size of the fine-tuning gain:**
  - Rust: large, 14-25%. Reth test 19-25%; 15 other Rust repos 15-21%.
  - Python, TypeScript, Go and C++: 0-3% on held-out repos, and at most 1.4% on Add.
  - Unverified explanation: the base model is weakest at Rust and has probably seen these popular repos in other
    languages.

## Caveats

- **Rust's data design differs.** It used a single repo and package split, and its benchmark was looked at once
  before its final run. A like-for-like multi-repo Rust rerun was considered and skipped.
- **One seed per arm**, one model size (6.7B), one LoRA recipe. Hyperparameters were not tuned for either arm, and
  nothing was tuned on test sets.
- **Over-length examples:** 3 Edit examples (1 Go, 2 C++) exceeded the 16,384-token context. They were skipped
  identically for all three models and are listed in the summaries. No example was truncated.
- **Perplexity is not completion quality.** No generated completions were scored in this experiment.
- **Unplanned reboots:** the C++ final evaluation was interrupted by two machine reboots and rerun from complete
  per-model outputs. The base results scored before the first reboot were complete and kept.

## Takeaway for practice

Random FIM holes are at least as good as AST-node holes for this kind of fine-tuning, and they are simpler, so use
random holes. Large gains from fine-tuning appeared only where the base model was weak (Rust).
