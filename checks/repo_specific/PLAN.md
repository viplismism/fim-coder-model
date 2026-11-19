# Plan: can a small model fine-tuned on one repo beat Copilot-class autocomplete on that repo? (DRAFT, no GPU)

## Question

For a "private-style" repo, take a 6.7B code model fine-tuned only on that repo's code written before a cutoff
date. On code written after the cutoff, does it complete better than (a) the same model untuned and (b) strong
off-the-shelf FIM models standing in for Copilot?

## 1. Repos ("private-style" = no model can have seen them)

- **Created recently.** The repo is created after the release of every model in the comparison (from 2025-01
  onward), so neither the base nor the stand-ins could have trained on any of its code. This is the closest public
  stand-in for private code.
- **Enough history on both sides of the cutoff:**
  - before the cutoff: >= 1 MB of target-language code;
  - after it: >= 150 non-merge commits that add target-language code.
- **Not derived from older code:** not a fork, not a rename or import of an older project (first commit date close
  to the creation date), and no large vendored or generated directories (excluded by path).
- **Same licence and owner rules as before.** Not in Real-FIM-Eval, and not one of our earlier training or held-out
  repos.
- **Proposal:** 3 repos, one each in Python, TypeScript and Go (languages the base model is strong in, so a gain
  would be repo-specific rather than "learning the language"). Each repo is its own experiment; one repo alone would
  be an anecdote.

## 2. Cutoff and data

- **Cutoff date C per repo** (e.g. 2026-05-01), with the test window C up to C + 6 months.
- **Training snapshot:** the default branch at the last first-parent commit with committer date < C. Only that tree
  is used for training.
- **Validation:** code added in the last ~4 weeks before C (from the pre-cutoff history), for sanity and loss
  curves only. Nothing is tuned on test.
- **Training recipe:** the same as before (6.7B, LoRA r16, lr 1e-4, 1 epoch, seed 42), with random holes only (AST
  lost). Holes are sampled from the snapshot's files, up to 7,549 pairs. A small repo yields fewer pairs; the number
  is reported, never padded.

## 3. Test set: real code written after the cutoff

- **Commits:** every non-merge first-parent commit (or merged PR) on the default branch inside [C, C + 6 months].
- **One example per added hunk** in a target-language file, built like Real-FIM-Eval Add. The file is taken at that
  commit; the hole is the added lines; the prefix and suffix are the rest of the file. Context is in-file only, the
  same for every model.
- **Filters:**
  - skip vendored, generated and lock files;
  - skip bulk commits (> 20 files or a reformat);
  - skip holes over 30 lines or under 1 non-blank line;
  - at most 5 examples per commit, so one big commit cannot dominate.
- **Target:** >= 500 examples per repo.

## 4. Leakage checks (all reported)

1. **Dates:** each test commit must have author and committer dates >= C, and must not be an ancestor of the cutoff
   commit (`git merge-base --is-ancestor`). This catches rebased or back-dated commits. Count and drop violators.
2. **Content:** drop an example if its hole text (whitespace-normalised, >= 40 chars) or any 5-line window of it
   already exists anywhere in the training snapshot (moved, copied or reverted code). The share of dropped examples
   is reported.
3. **Model cutoffs:** the repo's creation date and first commit must come after the release of every compared model.
   Check GitHub `created_at` and the oldest commit.
4. **No future files in training:** checked by hashing every training file against the snapshot tree.

## 5. Models

- **Base:** DeepSeek-Coder-6.7B base, no fine-tune.
- **Ours:** the same model with a repo-specific LoRA, one per repo.
- **Copilot stand-ins (proposal):**
  - **Qwen2.5-Coder-32B base:** the strongest widely recommended open FIM model.
  - **Codestral-22B v0.1:** strong at FIM, but its non-production licence means research use only, and the weights
    are gated on Hugging Face (accept the terms there first).
  - **Fallback for either:** Qwen2.5-Coder-7B, the same size as ours.
- **Prompts:** each model gets its own documented FIM format, and no prompt names the language.
- **Context:** Copilot itself also uses cross-file context; this plan gives every model in-file context only. A
  cross-file variant could be a later step.

## 6. Metrics and decision rule (fixed before any run)

- **Primary:** exact match of the greedy completion against the real added code (whitespace-normalised). Each model
  stops at its own end-of-infill token, or at 256 new tokens.
- **Secondary:**
  - edit similarity;
  - first-line exact match;
  - character perplexity (teacher-forced, tokenizer-independent).
- **Verdict per repo:** ours beats a stand-in if (EM_ours - EM_standin) > 0 and the 95% CI lower bound > 0 (paired
  bootstrap over commits). Ours vs base uses the same rule.
- **Headline:** the number of repos (out of 3) where ours beats each stand-in.

## 7. Compute (estimates)

- **Fine-tuning:** a few hours per repo, depending on pair count.
- **Generation:** 5 models x 3 repos x ~500 examples. The 32B model is the slowest, roughly 2-4 h per repo on the
  GB10. Total about 1.5-2 days of GPU.
- **Downloads:** about 110 GB (32B + 22B), with 3.3 TB free.

## 8. Order of work, with stop points

1. Repo search (CPU, GitHub metadata), then a shortlist for review. STOP.
2. Clone, pick C, build the snapshot, validation and test sets, run the leakage checks, report counts. STOP.
3. Download the stand-ins, then a smoke test of each model's FIM format on 10 validation examples. STOP.
4. GPU: fine-tune 3 adapters, then generate and score all models, then summarise.
