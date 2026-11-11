# Multi-language plan: AST-aware vs random holes, diverse data, 6.7B

Status: **plan agreed in discussion (2026-11-11), not started.** Nothing below has been built or trained. Steps that
produce something for review (repo lists, data counts) stop for approval before the next step.

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
| Recipes | **R, S, M** per language (A dropped: M contains aligned-span and is the paper's actual recipe) |
| Training per run | the same as the Rust run that worked: about **7,500 examples**, 1 epoch, identical settings for every run |
| Order | **one language at a time** (R, S, M, then scoring), so each language finishes before the next starts |

What dropping A loses: only "is aligned-span alone better than the mix?" (A vs M). Still answered: AST vs random
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

## 4. Recipes (identical rules in every language)

Every arm: **25% plain left-to-right examples** (2,048-token windows, identical across arms); same files, same slots,
same number of examples; hole sizes from tiny to very large; context ≤ 4,096 tokens incl. BOS/FIM/EOS; answers never
truncated; a slot is dropped from all arms if any arm's example does not fit; common random numbers across arms.

| Arm | FIM holes (75% of examples) |
|---|---|
| **R** (control) | 100% random character spans, each matched in file and length to the M arm's hole |
| **S** | 90% single-node + 10% random |
| **M** (paper's recipe) | 45% single-node + 45% aligned-span + 10% random |

- Single-node: one eligible named node, probability proportional to byte size; comments included; the root and
  keyword/punctuation leaves excluded.
- Aligned-span (length-matched, as in `screen-v3`): random center, target length = the slot's single-node hole length,
  snapped to the best-IoU run of direct named children of the smallest containing node.
- Results reported per hole-size class (tiny / small / medium / large / very large).

## 5. Data per language (diverse)

**Repo selection by a fixed rule, written down before cloning** (as for `heldout-v1`):
- GitHub search per language: permissive licence (MIT / Apache-2.0 / BSD), not a fork, not archived, sorted by stars.
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

**Validation (the only data used for decisions):**
- Character-level perplexity per language on fixed sets: single-node holes, random holes, aligned holes, left-to-right
  windows. Paired bootstrap 95% CIs, by example and by repo.
- Structural metrics from greedy generation on a fixed 150-pair subset per language: parse-clean after splicing,
  bracket balance, exact match, edit similarity, stop rate; plus the cheap syntax check from section 3.
- Per language, per repo, per size class.

**Winner rule per language (S vs R, M vs R), fixed before any run:**
1. arm − R ≤ 0 on single-node, random and aligned validation holes, with the CI excluding 0 on at least one;
2. left-to-right not worse than R by more than 0.005;
3. parse-clean, bracket balance and exact match not worse than R by more than 2 points (CI lower bound ≥ −0.02).

**Final, once per language, after its runs:** Real-FIM-Eval (Add; Edit provisional) and the language's locked held-out
set, for base, R, S and M. Every arm reported regardless of outcome.

## 9. Time

| Part | Estimate |
|---|---|
| Training: 5 languages × 3 recipes = 15 runs × about 11.5 h | about 7.2 days |
| Scoring: 20 models (base + R/S/M per language), merged adapters | about 1.3 days |
| **Total** | **about 8.5–10 days**; a finished language about every 1.7 days |

## 10. Order of work

1. Code changes + tests for the language configs (no data, no training).
2. Repo selection per language (rule recorded, commits pinned) → **review the repo lists**.
3. Overlap checks → report.
4. Build data per language (R/S/M + validation + held-out) → **review counts and size classes**.
5. Per language, in order Rust, Python, TypeScript, Go, C++: train R, S, M → validation scoring → final evaluation → report.
6. Final write-up across all languages.
