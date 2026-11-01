# Near-duplicate check (datatrove MinHash)

Script `neardup.py`, settings `settings.json` (with rationale), full output `report.json`.
Inputs are the same pins as `checks/overlap/`: reth `84caec584e79ddef33572394f787d2d1203a23c7`,
`gonglinyuan/real_fim_eval` `a36062544c7ed6c4e5ffb8dad8536fc7777e1f36` (Rust Add + Edit), and the
crate split recorded in `data/<version>/split.json`. Raw pinned sources are not modified or filtered.

## Method

- **Tokens and normalization (Rust):** identifiers, numbers and single punctuation characters; whitespace
  ignored; case kept; numbers normalized; no punctuation removal or diacritic folding.
- **Shingles:** 10-token n-grams (about one line of Rust), so a single generic line is not enough to match.
- **Units:** whole files, plus 30-non-blank-line chunks (stride 15) on both sides; benchmark answers of 5+
  non-blank lines as their own unit; Edit examples give both pre- and post-change files. Identical benchmark
  texts (same file in several examples) are one document.
- **datatrove:** `MinhashDedupSignature` → `MinhashDedupBuckets` (14 bands × 8 rows: a pair with Jaccard
  ≈0.72 has a 50% chance of becoming a candidate) → `MinhashDedupCluster`.
- **Verification:** every benchmark document that lands in a cluster with any reth document is compared
  exactly (shingle Jaccard and containment) with its best reth partner; Jaccard ≥ 0.5 is then inspected by
  hand. The same is done for reth documents in different splits.
- **Positive control:** 20 benchmark files are copied to the reth side with numbers changed, indentation
  re-flowed and ~10% of lines deleted, chunked like real reth files. They must be found.

## Results

| | |
|---|---|
| Documents | reth: 1,314 files + 28,165 chunks; benchmark: 4,527 post-change files, 1,950 pre-change files, 90,556 chunks, 2,876 answers (64 reth / 93 benchmark docs too short to shingle) |
| Positive control | 19 / 20 found. The miss had file-level Jaccard 0.465 to its source; found controls go down to 0.646 |
| **Benchmark vs reth** | **0 benchmark documents in any cluster with a reth document → nothing to verify** |
| reth cross-split (MinHash) | 3 candidate pairs, 2 with Jaccard ≥ 0.5 (test/train 0.715, test/validation 0.575) |
| Those 2 pairs, inspected | both are the crate-root `lib.rs` header every reth crate carries (`#![doc(html_logo_url …)]`, `#![cfg_attr(...)]`): repository boilerplate, not copied logic. Not filtered. |
| reth cross-split (verbatim 5-line windows) | validation: 64 of 23,909 windows also in train (0.27%); test: 52 of 24,929 (0.21%); validation∩test: 35 |
| Exact duplicate files across splits | 0 |

## Coverage and limitations

"No near-duplicate detected" is not proof that none exists. This check would miss:

- copies more altered than the control (file Jaccard below ~0.65 on these shingles), e.g. heavy renaming;
- a short copied fragment inside a long, otherwise different document (MinHash estimates Jaccard, not
  containment; the 30-line chunks and the exact 5-line window check in `checks/overlap/` reduce this gap);
- benchmark answers shorter than 5 non-blank lines as standalone units (they are still inside the file and
  chunk units);
- code that was in older reth commits but not in the pinned snapshot.

Within reth, crates are disjoint across splits, but crates in one repository share conventions (headers,
imports, test scaffolding), so some stylistic similarity across splits is expected and not removed.
