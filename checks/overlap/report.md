# Overlap check: reth vs. Real-FIM-Eval (Rust)

Generated 2026-10-31T21:38Z by `run.sh` → `overlap_check.py`, with the settings in `settings.json`.
The full machine-readable output is in `report.json`.

## Inputs

| | |
|---|---|
| reth | `paradigmxyz/reth` @ `84caec584e79ddef33572394f787d2d1203a23c7` (2026-10-30T21:07:22Z, `main` HEAD at time of check), 1,326 `.rs` files |
| Benchmark | `gonglinyuan/real_fim_eval` @ `a36062544c7ed6c4e5ffb8dad8536fc7777e1f36` |
| `add/test.jsonl.gz` sha256 | `bd2ae46588c83c79bb4f99f967700377c4440c3607ad8550cb0334eb8c8a3f22` |
| `edit/test.jsonl.gz` sha256 | `b4f19e8f40535b6421b7c847eebfb64c0be28aa59dc6f636e13fc6173a5b29c5` |
| Rust subset | 4,727 examples (2,747 add, 1,980 edit) from 49 source repos |

## Result: no overlap detected

| Check | Result |
|---|---|
| Source repo name matches `reth\|paradigmxyz\|op-rs\|op[-_]reth` | none |
| Fork/rename status of the 49 source repos (GitHub API) | not recorded in this run: unauthenticated rate limit. An earlier, unsaved run the same day (2026-10-31 UTC) found no forks; the only rename was `facebook/relay` → `meta/relay`. Rerun with `GITHUB_TOKEN=... bash run.sh --github_api` to record it. |
| Paths/code mentioning `reth`, `reth_*`, `reth-*`, `paradigmxyz`, `op-reth`, `revm` | 0 examples |
| Code mentioning `alloy` (a library reth depends on, not reth code) | 84 examples: `unionlabs/union` 78, `linera-protocol` 6 |
| 5 consecutive normalized lines (≥150 chars) shared with reth | 0 examples |
| Middle lines ≥60 chars found verbatim in reth | 64 hits, 12 unique lines, all generic Rust boilerplate (`fn fmt(...)`, `fn serialize<S>(...)`, `#[derive(...)]`) |
| Positive control: detector finds same-repo reuse in zed | 51 zed matches, 0 false matches from other repos |

## What this does and does not show

"No overlap detected" means none of the checks above found a match. It is not proof
that no reth-derived code is in the benchmark. The checks would miss:

- copied code that was renamed, reformatted beyond per-line whitespace, or shorter than 5 non-blank lines;
- code that came from an older reth commit and has since been removed from reth;
- reth code vendored or ported by a project under another name, if it was changed;
- forks, unless the GitHub check above is rerun and recorded.

Rerun whenever either pin in `settings.json` changes.
