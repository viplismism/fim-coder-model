# Next steps (from Vipul)

Decisions on your four questions:
1. Sanity check: approved, but run it AFTER step 0 below.
2. Left-to-right share: 25%, for every arm including the control.
3. Controls: a shared random control (matched to M) for 1.3B screening, then per-variant length-matched controls for the 6.7B confirmation.
4. M+C: skip it. n=24 is too small, and boosting comments is tuning, not a hypothesis.

## Step 0, before any training
Add structural metrics to the scorer and rescore the EXISTING 6.7B base/AST/random generations. Validation only, don't touch test, the benchmark or held-out.
- Parse rate: splice each completion into its file and check with tree-sitter-rust whether the file parses with no ERROR/MISSING nodes (compare against the original file's parse state).
- Compile rate: for the subset where it's practical, splice into the pinned reth checkout and run `cargo check` on that crate. Report coverage (how many examples could be checked).
- Also report exact match, edit similarity, and the rate of balanced brackets/blocks.
- Report AST vs. random and each vs. base, broken down by hole set (AST holes vs. random holes) and boundary type, with paired bootstrap 95% CIs and n per bucket.
- Generate enough examples for the CIs to mean something (all validation pairs if time allows, not 40).

Then tell me whether AST shows any structural advantage. After that, start the 1.3B sanity check and add the structural metrics to the screening winner criteria. Write down the final variant list (R, S, A, M) and the rules before running anything, and report every arm.
