# Screening plan (declared before any screening run)

Written 2026-11-03, before the 1.3B sanity check and before any screening arm is trained. Every arm listed here
is reported, whatever its result. Test split, Real-FIM-Eval and the locked held-out set (`data/heldout-v1`) are not
used for any decision below; they are scored once, in the final 6.7B confirmation.

## 0. Sanity check (1.3B): does 1.3B reproduce the 6.7B direction?

- Model `deepseek-ai/deepseek-coder-1.3b-base` @ `c919139c3a9b4070729c8b2cca4847ab29ca8d94` (same tokenizer and PSM
  format as 6.7B, verified). bf16 base, plain AdamW, LoRA r16/alpha32/dropout 0.05 on all 7 projections.
- Data `rust-fim-v2`, unchanged. Arms S0 (single-node, as in the 6.7B run) and R0 (matched random), no hygiene.
- Identical training settings to the 6.7B runs (1 epoch = 472 updates, lr 1e-4 cosine, warmup 24, seed 42).
- Scored on full validation: AST holes (518), random holes (518), left-to-right (136 windows).
- **6.7B direction** (validation): AST holes tie (−0.0000 [−0.0004, +0.0004]); random holes R better
  (+0.0021 [+0.0016, +0.0027]); left-to-right R better (+0.0048 [+0.0039, +0.0057]).
- **Pass** if on 1.3B: random holes S0 − R0 > 0 with CI excluding 0, AND AST holes S0 − R0 CI does not lie entirely
  below 0, AND left-to-right S0 − R0 ≥ 0 (point estimate). If the random-hole CI includes 0, run seed 43 for both arms
  and pool before deciding. **Fail** → no screening on 1.3B; report and decide with you how to proceed.

## 1. Arms (1.3B screening, only if the sanity check passes)

Hygiene for **every** arm, including the control:
- 25% of training examples are plain left-to-right windows from the same train files (identical L2R examples in
  every arm, same seed).
- Within the FIM examples of AST arms, 10% are random holes (length-matched to that arm's own AST holes, same file).

| Arm | FIM examples (75% of the training set) |
|---|---|
| **R** (control) | 100% random holes, each length- and file-matched to an M-arm hole (shared control) |
| **S** | 90% single-node holes + 10% random |
| **A** | 90% aligned-span holes + 10% random |
| **M** | 45% single-node + 45% aligned-span + 10% random |

- Single-node: as now (one named node, probability proportional to byte size; comments included; root and
  keyword/punctuation leaves excluded).
- Aligned-span (paper's second method): sample a character span with uniform random endpoints; take the smallest
  named node containing it; choose the contiguous run of that node's direct named children with the largest
  character-level IoU with the span; the hole is that run's exact byte range.
- Same number of training examples per arm, same files, same context-window rule (≤ 4,096 tokens incl. BOS/FIM/EOS),
  answers never truncated, invalid matched pairs dropped together. Same training settings as the sanity check.
- Seeds 42 and 43 for every arm: 8 runs.

## 2. Evaluation for screening (validation only, identical examples for every arm)

- Perplexity (teacher-forced, char-level) on fixed validation sets: AST holes (518), random holes (518),
  aligned-span holes (new fixed set from the same validation files), left-to-right windows (136).
- Structural metrics from greedy generation on a fixed validation subset (pairs with answers ≤ 512 tokens, the same
  subset for every arm): parse-clean after splicing, bracket-balance match, exact match, edit similarity, stop rate.
- Paired bootstrap 95% CIs (resampling examples), seeds pooled; per-seed results reported too.

## 3. Winner rule (each AST arm vs R)

An arm is a **winner** only if ALL hold:
1. Perplexity: arm − R ≤ 0 on AST holes, random holes and aligned-span holes for **both** seeds, and the pooled CI
   excludes 0 (arm better) on at least one of these sets.
2. Left-to-right non-inferiority: pooled arm − R ≤ +0.005.
3. Structural non-inferiority: pooled parse-clean rate and bracket-balance match not worse than R by more than
   2 percentage points (lower CI bound of arm − R ≥ −0.02), and pooled exact match not worse (CI lower bound ≥ −0.02).
4. The direction of (1) is the same for both seeds.

If several arms win, the one with the largest pooled perplexity gain on random holes is confirmed first. If none wins,
the result is "no AST variant beat the random control" and every arm is still reported.

## 4. Confirmation (6.7B, after screening)

Winner(s) and **per-variant length-matched random controls**, 3 seeds each, denser data. Final evaluation once on
reth test, Real-FIM-Eval Rust (Add; Edit provisional) and the locked held-out set.
