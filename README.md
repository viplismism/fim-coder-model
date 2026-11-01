# Rust FIM fine-tuning experiment

**Question:** does AST-based fine-tuning improve a small coding model's Rust infilling on unseen code?
Improvement is not assumed; the measured comparison is the result.

**Design:** DeepSeek-Coder-6.7B base (4-bit NF4) + LoRA adapter, PSM only. Two arms built from the same
reth files with the same pipeline: **AST** single-node holes (one tree-sitter node per example, chosen with
probability proportional to its byte size) and **random** holes matched to each AST hole's file and
character length. Primary comparison is against the same 4-bit base without an adapter.

## Layout

```
configs/experiment.yaml   every setting; `null  # TO DECIDE` values must be set before training
fimexp/                   pipeline: sources/pins, PSM format, crate split, holes, data build, trainer, scorer
tests/                    data, tokenization, label-mask and scoring tests (pytest)
checks/overlap/           reth vs Real-FIM-Eval exact/mention overlap check (pinned snapshots)
checks/neardup/           datatrove MinHash near-duplicate check (benchmark vs reth, across reth splits)
checks/env/               CUDA / 4-bit / LoRA / tokenizer round-trip checks
env/                      how .venv was built, pinned lock file
```

Not committed (see `.gitignore`): `.venv/`, `sources/` (pinned reth checkout + benchmark files),
`.cache/` (model), `data/` (generated datasets), `runs/`.

## Pinned inputs

| | |
|---|---|
| reth | `paradigmxyz/reth` @ `84caec584e79ddef33572394f787d2d1203a23c7` |
| Benchmark | `gonglinyuan/real_fim_eval` @ `a36062544c7ed6c4e5ffb8dad8536fc7777e1f36` (Rust Add + Edit) |
| Model | `deepseek-ai/deepseek-coder-6.7b-base` @ `ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912` |

## Commands

```bash
.venv/bin/python -m pytest -q tests/
.venv/bin/python -m fimexp.build_data --version <name>            # writes data/<name>/, never overwrites
.venv/bin/python checks/neardup/neardup.py --split_json data/<name>/split.json
.venv/bin/python -m fimexp.train --data_version <name> --arm ast --run_name <run>      # refuses while settings are undecided
.venv/bin/python -m fimexp.score --tasks bench_add,bench_edit --out runs/eval/<name> [--adapter runs/<run>/final_adapter]
```

Development evaluation reads the validation split. The test split and the benchmark are for final
reporting only and never select settings or checkpoints (`score.py` requires `--final` for test).

## Prior pilot

`viplismism/deepseek-coder-6.7b-fim-reth-v1` and the `reth-fim-dataset` it was trained on are historical
only: unknown reth commit, item-level spans, and its checkpoint was selected on its own test split.
They are not inputs to this experiment.

## License

MIT
