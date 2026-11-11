# Multi-language plan: AST-aware vs random holes, diverse data, 6.7B

Status: **step 1 (code + tests) done 2026-11-11; nothing built or trained.** Approval is per stage and per language:
the first language is chosen by you, and **the full five-language training run is not approved yet**. Nothing is
pushed or merged without an explicit go.

## 1. Question

Does AST-aware hole selection beat matched random holes when fine-tuning a code model, **across the Real-FIM-Eval
languages and with varied training data**?

Rust result so far (6.7B, one repo, single-node holes, seed 42): fine-tuning helps a lot (14–21% less uncertainty per
character on validation, the reth test, Real-FIM-Eval Rust and 15 fresh repos), but AST holes do not beat random
holes. This round removes the two biggest limits of that result: only half of the paper's recipe was tested, and the
training data came from a single repo.

## 2. Decisions

| Item | Decision |
|---|---|
| Model | **DeepSeek-Coder-6.7B base only** (bf16, frozen) + LoRA adapter; no 1.3B runs |
| Languages | **Rust, Python, TypeScript, Go, C++**: only languages Real-FIM-Eval covers as its own subset (C excluded) |
| Final test | **Real-FIM-Eval** per language (Add; Edit labelled provisional) + a **fresh locked held-out set** per language; used once, never for training or choosing settings |
| Recipes | **R, S, M** per language (A dropped). **M is a paper-style mix, not an exact reproduction** of the paper's recipe |
| Training per run | the same as the Rust run that worked: about **7,500 examples**, 1 epoch, identical settings for every run |
| Order | **one training run at a time**; one language at a time (R, S, M, then scoring); you approve each stage |
| First language | **Python** for the 6.7B pilot (largest benchmark, not used in earlier experiments); proposed by you, pending final go |
| Main comparison | **to be fixed before any run** (proposal: **M vs R**, the exactly length-matched paper-style mix vs random); all others secondary |

What dropping A loses (M is the paper-style mix): only "is aligned-span alone better than the mix?" (A vs M). Still answered: AST vs random
(S vs R, M vs R), the paper's recipe vs random (M vs R), and whether aligned-span adds value on top of single-node
(M vs S).

## 3. Languages

| Language | Real-FIM-Eval (Add + Edit) | Benchmark repos | Structural check |
|---|---|---|---|
| Rust | 4,727 (2,747 + 1,980) | 49 | tree-sitter parse; `cargo check` where practical |
| Python | 6,271 (3,687 + 2,584) | 75 | tree-sitter parse; `py_compile` |
| TypeScript | 3,182 (1,671 + 1,511) | 32 | tree-sitter parse |
| Go | 2,587 (1,312 + 1,275) | 18 | tree-sitter parse; `gofmt -e` |
| C++ | 3,265 (1,798 + 1,467) | 39 | tree-sitter parse (compiling needs headers/build systems) |

Other Real-FIM-Eval languages (Java, Ruby, C#, JavaScript, Kotlin, PHP, Scala) can be added later with the same pipeline.

- TypeScript: Real-FIM-Eval contains only `.ts` files, so training also uses `.ts` only (TypeScript grammar).
- C++: the 315 `.c` files inside Real-FIM-Eval's cpp subset are excluded (C is not a Real-FIM-Eval language), leaving
  2,950 C++ examples (1,643 Add + 1,307 Edit).

## 4. Recipes (identical rules in every language)

Every arm: **25% plain left-to-right examples** (2,048-token windows, identical across arms); same files, same slots,
same number of examples; hole sizes from tiny to very large; context ≤ 4,096 tokens incl. BOS/FIM/EOS; answers never
truncated; a slot is dropped from all arms if any arm's example does not fit; common random numbers across arms.

| Arm | FIM holes (75% of examples) |
|---|---|
| **R** (control) | 100% random character spans, each matched in file and length to the M arm's hole |
| **S** | 90% single-node + 10% random |
| **M** (paper-style mix) | 45% single-node + 45% aligned-span + 10% random |

- Single-node: one eligible named node, probability proportional to byte size; comments included; the root and
  keyword/punctuation leaves excluded.
- Aligned-span (length-matched, as in `screen-v3`): random center, target length = the slot's single-node hole length,
  snapped to the best-IoU run of direct named children of the smallest containing node.
- Results reported per hole-size class (tiny / small / medium / large / very large).

**Separate adapters (confirmed 2026-11-11).** Every arm is its own training run: a **fresh LoRA adapter
starting from the same untouched base model**, trained once on that arm's data. R, S and M are never trained in
sequence through one adapter (not R then S then M). Comparisons are between independently trained adapters.

**How fair each comparison is (measured on the Rust `screen-v3` construction):**
- **M vs R is length-matched:** R's hole has the same file and length as M's in 100% of slots.
- **S vs R is not fully matched:** same length in only 50% of slots; S answers are shorter (median 277 vs 321
  characters, mean 1,162 vs 1,308) and S has about 15% fewer answer tokens to learn from (2.50 M vs 2.95 M).
  So **S vs R compares whole recipes, not syntax boundaries alone**, and is reported that way. Optional fix (adds one
  run per language): a separate random control R_S length-matched to S. **Decision: R_S is worth adding to test whether
  syntax-aligned holes help (S vs R_S); it is not started yet and will be scheduled as its own approved run.**

## 5. Data per language (diverse)

**Repo selection by a fixed rule, written down before cloning** (as for `heldout-v1`):
- GitHub search per language: permissive licence (MIT / Apache-2.0 / BSD), not a fork, not archived, sorted by stars,
  primary language = the language. **Training pool: created before 2024-01-01, pushed since 2025-01-01, ≥ 2,000 stars**
  (established projects; a clean time split from the 2024–25 held-out repos). **Held-out: created 2024-01-01 to
  2025-06-30, ≥ 500 stars.** Lists, tutorials, prompt/workflow/skill collections and algorithm compilations excluded.
  Proposals: `checks/multilang/proposed_repos_v2.md` (rule: `propose_repos.py`; v1 kept for transparency).
- **Excluded:** every Real-FIM-Eval repo (all 186, any language), every repo in any held-out set, at most one repo per
  owner; generated/vendored files and files > 300 KB.
- A size window per repo plus a per-repo cap on examples, so no repo dominates (important for C++).
- **Training pool:** 12 repos per language, split by repo into 10 train / 2 validation (seeded, recorded).
- **Fresh held-out test:** 10 repos per language created 2024-01 or later, 100 matched pairs per repo, locked,
  scored once. Rust gets a new `heldout-v2` (`heldout-v1` was already used once).
- **Overlap checks before training:** exact 5-line windows + datatrove MinHash with planted controls, between the
  training pool and (a) the language's Real-FIM-Eval subset, (b) the held-out repos, (c) train vs validation. Any
  meaningful match stops the build for review.

## 6. Code changes (before any data is built)

- Per-language config: tree-sitter grammar, comment node types, keyword list, package detection (Rust `Cargo.toml`,
  Python `pyproject.toml`/`setup.py`, TypeScript `package.json`, Go `go.mod`, C++ `CMakeLists.txt`/top-level directory),
  file extensions (C++: `.cpp .cc .cxx .hpp .hh .h`).
- Grammars `tree-sitter-python`, `tree-sitter-typescript`, `tree-sitter-go`, `tree-sitter-cpp`, pinned, in `.venv` only.
- Multi-repo data builder; tokenizer whitespace round-trip check per language.
- Tests per language: eligibility rules, UTF-8 offsets, exact reconstruction, aligned-span boundaries.
- Scorer: aligned-hole validation set; Real-FIM-Eval per language; scoring with the adapter **merged** into a copy of
  the bf16 base (same model, about 2× faster), applied identically to every arm.

## 7. Training

- Same settings as the Rust 6.7B runs: bf16 base, plain AdamW, LoRA r16 / alpha 32 / dropout 0.05 on all 7
  projections, lr 1e-4 cosine, warmup 5%, 1 epoch, seed 42, final checkpoint, one adapter per language.
- Per run: about 7,500 examples ≈ 470 updates ≈ 11.5 h. Logged locally (`logs/`) and to the private W&B project.
- One seed per arm. A later multi-seed study, if any, gets its own declared plan.

## 8. Evaluation

**Kept as they are:** the earlier Rust evaluations, scorer, prompts (PSM format, inferred Edit layout) and results, so
new runs are directly comparable. Real-FIM-Eval stays alongside the new tests. Other languages use **the same method**
with their own language-specific examples. **Disclosure:** Rust Real-FIM-Eval has already been used once (for the 6.7B
seed-42 models); nothing was tuned on it. The new locked held-out sets (including Rust `heldout-v2`) are fresh.
No setting is ever chosen or changed using final-test results.

**Primary score per comparison (approved 2026-11-11; frozen before any run):** character-level perplexity on the language's
**Real-FIM-Eval Add** subset (the paper's metric on real developer additions, neutral to all hole recipes), arm − R,
with a paired bootstrap 95% CI resampling **benchmark repos** (examples within a repo are correlated).
- **Winner rule:** arm X beats R in language L if and only if the point estimate of X − R is below 0 **and** the
  upper bound of its 95% repo-level CI is below 0. Otherwise: "no evidence X beats R" (with the estimate and CI).
- **Main comparison: chosen and written here before the first run, never after seeing results** (proposal: M vs R).
  Secondary: S vs R (whole-recipe comparison), S vs R_S if R_S is run (syntax-aligned holes, length-matched), M vs S.
  Secondary results are reported with the number of comparisons in mind; the earlier Rust tests are reported alongside.

**Reported too, never used to decide (including negative results):**
- Real-FIM-Eval Edit (provisional layout), the locked held-out set (single-node holes, random holes, left-to-right),
  validation perplexity (single-node, random, aligned, left-to-right; per repo and size class).
- **Generated-code checks (150 pairs per language) are descriptive only.** A paired sample-size check on the Rust
  structural data gives 95% CI half-widths of about ±5–6 points (parse-clean, bracket balance) and ±3–4 points (exact
  match) at n = 150; a ±2-point tolerance would need about 1,000–1,450 pairs. So no structural tolerance is used to
  pick winners.
- Validation has 2 repos per language, so validation CIs are by example only (monitoring, not decisions).

## 9. Time

| Part | Estimate |
|---|---|
| Training: 5 languages × 3 recipes = 15 runs × about 11.5 h | about 7.2 days |
| Scoring: 20 models (base + R/S/M per language), merged adapters | about 1.3 days |
| **Total** | **about 8.5–10 days**; a finished language about every 1.7 days |

## 10. Order of work

1. Code changes + tests for the language configs (no data, no training). **Done 2026-11-11.**
2. Repo selection per language (rule recorded, commits pinned) → **review the repo lists**. v1/v2 not approved; v3 rule
   (`propose_repos_v3.py`): one repo per owner across all languages and lists, same-name benchmark repos excluded (e.g.
   `Eugeny/tabby`, same name as Real-FIM-Eval's `TabbyML/tabby`), target-language code 0.3–30 MB and ≥ 50% of the repo.
3. Overlap checks → report.
4. Build data per language (R/S/M + validation + held-out) → **review counts and size classes**.
5. For the language you choose first: confirm the separate-adapter setup, then train R, S, M one run at a time →
   validation scoring → final evaluation → report. Each next language is approved separately.
6. Final write-up across all languages.
