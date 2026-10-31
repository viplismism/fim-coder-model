# FIM Coder Model

**Question:** does AST-based fine-tuning improve a small coding model's Rust
infilling on unseen code?

Improvement is not assumed. The result is the measured comparison between the
fine-tuned model and its base, on code held out from training.

## Experiment (draft — open items marked TODO)

| | |
|---|---|
| Base model | `deepseek-ai/deepseek-coder-6.7b-base` |
| Method | QLoRA (4-bit NF4, LoRA r=64 on all attention + MLP projections) |
| Training data | FIM samples whose middle is an AST node (function, function body, impl, struct, enum, trait, const, use) from Rust source |
| Source corpus | [reth](https://github.com/paradigmxyz/reth) — TODO: pin a commit |
| "Unseen code" | TODO: decide the held-out set (see below) |
| Metrics | exact match (pass@1), edit similarity; reported overall and by node type |
| Comparison | fine-tuned adapter vs. the same base model, same prompts, greedy decoding |

### Decisions to make before the first run

1. **What counts as unseen.** The current split holds out whole files from the
   same repo. Stricter options: a different Rust repo, or reth code written
   after the base model's training cutoff (the base model may have seen reth).
2. **Whether "AST-based" needs its own control.** Base vs. fine-tuned shows
   that fine-tuning helps; showing that *AST-based span selection* helps also
   needs a run fine-tuned on random-span FIM samples from the same files.
3. **Fixed eval set size and seeds**, and whether to report confidence
   intervals across samples.

## Pipeline

```
ast_extractor/   Rust: parse .rs files, emit AST node spans
datagen/         build prefix/middle/suffix FIM samples, file-level split
training/        QLoRA fine-tuning (config.yaml)
evaluation/      benchmark tuned vs. base on the held-out split
utils/           tokenizer whitespace guard (used by training and eval)
```

Training and eval are pinned to `transformers==4.49.0` + `trl==0.17.0`: newer
`transformers` silently corrupts code whitespace during tokenization, and both
scripts refuse to run if the tokenizer fails a whitespace round-trip check.

## Prior result (v1, before this experiment was defined)

Adapter [`viplismism/deepseek-coder-6.7b-fim-reth-v1`](https://huggingface.co/viplismism/deepseek-coder-6.7b-fim-reth-v1),
200 held-out-file samples: pass@1 31.0% tuned vs. 13.5% base. Same-repo
held-out files, no random-span control — treat as a pilot, not the result.

## License

MIT
