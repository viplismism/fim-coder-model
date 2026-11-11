# File-level decontamination rule (written before applying it)

Applies to every language. Written 2026-11-11, before the Python data is rebuilt.

**Scope.** Every file in the training pool (both the train and the validation repos). The test sets — the language's
Real-FIM-Eval subset and its locked held-out repos — are never modified.

**Comparison targets.** (a) Every Real-FIM-Eval example of the language: the post-change file (prompt + answer + suffix),
the pre-change file for Edit (prompt + removed code + suffix), and the answer itself. (b) Every file of the language's
held-out repos.

**Licence-header handling.** Before comparison, only a file's leading comment block is removed, and only if it contains
licence/copyright wording (copyright, licensed under, licence/license, SPDX-License-Identifier, permission is hereby
granted, all rights reserved). All other comments are kept. Matches explained entirely by such a header are logged as
licence-header exceptions, not removals.

**A training-pool file is removed if any of these holds against any target:**
1. **Exact window:** it shares at least one exact 5-line window (whitespace-normalised, blank lines dropped, ≥ 150
   characters) that contains no licence wording.
2. **Near-duplicate, whole file or 30-line chunk:** a datatrove MinHash candidate with verified shingle Jaccard ≥ 0.5,
   computed after licence-header removal.
3. **Near-duplicate, 10-line chunk (finer check):** the same, on 10-line chunks (stride 5).

Removal is of the whole file, regardless of whether the shared code touches a scored answer.

**After removal.** The dataset is rebuilt with the unchanged recipe and settings; if clean data cannot reach the pair
targets (7,549 train, 518 validation), the shortfall is reported, never padded with duplicates. All overlap checks are then
rerun on the rebuilt data.

**Detector checks (planted copies).** 20 altered whole-file copies (as before, including the previously missed short
file) and 20 altered fragments (12–20 lines of a benchmark file spliced into the middle of a training-pool file) are
placed on the training-pool side. Each is reported as caught by the exact-window rule, the 30-line check, the 10-line
check, or missed.
