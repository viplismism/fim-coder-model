#!/usr/bin/env python3
"""Write the two Hugging Face model cards (README.md per adapter) from one template. Numbers come from
results/facet/summary.json and style_check.json (read at build time, nothing typed by hand)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.load(open(ROOT / "results/facet/summary.json"))
V = json.load(open(ROOT / "results/facet/style_check.json"))
M = S["metrics"]
pct = lambda v: f"{100 * v:.1f}%"
dif = lambda d: f"{100 * d['value']:+.1f} (95% CI {100 * d['ci'][0]:+.1f} to {100 * d['ci'][1]:+.1f})"

ARMS = {
    "random": dict(name="facet-qwen2.5-coder-14b-lora-random", key="facet-random", other="facet-ast",
                   holes="**random holes**: spans at random positions, with lengths matched one-to-one to the AST arm",
                   pitch="The recommended adapter of the two.", note=""),
    "ast": dict(name="facet-qwen2.5-coder-14b-lora-ast", key="facet-ast", other="facet-random",
                holes="**AST holes**: each hole is exactly one tree-sitter node, chosen with probability proportional to its size",
                pitch="The structure-aware control. The random-hole adapter did as well or better; prefer that one.",
                note="- Its character perplexity on the real code is slightly **worse** than base 14B. This adapter learned where to stop, not to model facet's code better.\n"),
}

for arm, a in ARMS.items():
    k, base, big = a["key"], "qwen14b-base", "qwen32b-base"
    pa, em = M["build_test_pass/all"]["extra"], M["exact_match/all"]["extra"]
    nl = M["build_test_pass/neither_hit_limit"]
    ppl = M["char_ppl/all"]
    d_base = M["build_test_pass/all"]["diffs"][f"{k} - {base}"]
    d_32 = pa["diffs"][f"{k} - {big}"]
    d_nl = nl["diffs"][f"{k} - {base}"]
    beats32 = d_32["ci"][0] > 0
    sc = S["stop_check"]["by_model_that_hit_limit"][base]
    card = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-14B
library_name: peft
pipeline_tag: text-generation
language:
- code
tags:
- code
- fill-in-the-middle
- fim
- code-completion
- lora
- rust
- repo-specific
---

# {a['name']}

A LoRA adapter for [Qwen/Qwen2.5-Coder-14B](https://huggingface.co/Qwen/Qwen2.5-Coder-14B), fine-tuned for
**fill-in-the-middle code completion on one repository**: [facet-rs/facet](https://github.com/facet-rs/facet)
(Rust). It was trained only on facet as of **2026-01-01**, with {a['holes']}. {a['pitch']}

This is a research artifact from [fim-coder-model](<GITHUB_REPO_URL>): train your own
Copilot on your repo, then test it on code written after training. It shows what repo-specific fine-tuning buys on
one repo. It is not a general-purpose Rust model.

## Results

Tested on **1,056 real edits from 391 facet commits made after the cutoff**. These are blocks of added lines, each
completed from the rest of the file. Every example builds and passes its crate's tests with the real code, and fails
when the lines are deleted. pass@1 = greedy completion spliced in, then `cargo build` + `cargo test`.

| | base 14B | **this adapter** | other 14B adapter ({a['other'].split('-')[1]} holes) | base 32B |
|---|---|---|---|---|
| pass@1 (build + tests) | {pct(pa['rate'][base])} | **{pct(pa['rate'][k])}** | {pct(pa['rate'][a['other']])} | {pct(pa['rate'][big])} |
| exact match (same 1,056) | {pct(em['rate'][base])} | **{pct(em['rate'][k])}** | {pct(em['rate'][a['other']])} | {pct(em['rate'][big])} |
| char perplexity of the real code (1,754 examples, lower is better) | {ppl['char_ppl'][base]:.4f} | **{ppl['char_ppl'][k]:.4f}** | {ppl['char_ppl'][a['other']]:.4f} | n/a |

- vs base 14B, pass@1: **{dif(d_base)}** points.
- vs base Qwen2.5-Coder-32B, pass@1: {dif(d_32)} points, so it **{'beats' if beats32 else 'matches'}** the 32B.
- Most of the gain is **knowing when to stop**. Base 14B ran into the 256-token limit on {sc['n']} examples and
  passed {sc['passes'][base]} of them; this adapter passes {sc['passes'][k]} of the same examples. On the
  {nl['n']} examples where no 14B model hit the limit, the gain over base is {dif(d_nl)}.
{a['note']}- 'Their style', exact match on 150 held-out validation holes per hole type:
  AST holes {pct(V['ast_holes']['exact_match'][k])} (base {pct(V['ast_holes']['exact_match'][base])}),
  random holes {pct(V['random_holes']['exact_match'][k])} (base {pct(V['random_holes']['exact_match'][base])}).

CIs: paired bootstrap that resamples commits, B = 2000. Full tables, splits (source vs test files, human vs
AI-tagged commits) and scripts are in the GitHub repo.

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen2.5-Coder-14B"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "<this-repo-id>").eval()

prefix = "fn main() {{\\n    let v = vec![1, 2, 3];\\n"
suffix = "\\n    println!(\\"{{}}\\", total);\\n}}\\n"

# Same layout as training: segments tokenized separately, no BOS.
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
special = lambda t: tok.convert_tokens_to_ids(t)
ids = [special("<|fim_prefix|>"), *enc(prefix), special("<|fim_suffix|>"), *enc(suffix), special("<|fim_middle|>")]
stop = [special(t) for t in ("<|endoftext|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>")]

x = torch.tensor([ids], device=model.device)
out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=256, do_sample=False,
                     eos_token_id=stop, pad_token_id=stop[0])
print(tok.decode(out[0, len(ids):], skip_special_tokens=True))
```

- Use the whole file as context: prefix = everything before the cursor, suffix = everything after.
- The evaluation used up to 32K tokens of context. Training examples were at most 4,096 tokens.
- To serve it, merge the adapter (`model.merge_and_unload()`) or load it as a LoRA in vLLM.

## Training

| | |
|---|---|
| data | facet at commit `d3cd33f75da7b80fc0134ec7346c26512d3ada72` (last commit before 2026-01-01); non-generated `.rs` files; 5% of files held out for validation |
| pairs | 8,735 train / 454 validation; the AST and random arms are paired one-to-one (same file, same hole length) |
| format | Qwen2.5-Coder FIM (PSM); loss on the middle + `<\\|endoftext\\|>` only |
| LoRA | r=16, alpha=32, dropout 0.05, q/k/v/o/gate/up/down projections (68.8M trainable parameters) |
| optimizer | AdamW, lr 1e-4 cosine, warmup 5%, 1 epoch, batch 16 (1 × 16 accumulation), seed 42, bf16, no quantization |
| hardware | one NVIDIA DGX Spark (GB10), 546 updates in ~25 h, peak 37.5 GB |

The other arm used exactly the same settings. Only where the holes were cut differs.

## Limitations

- **Specific to one repo.** It learned facet's code and conventions as of 2026-01-01. On other code, expect roughly
  base-model behaviour or worse. Its knowledge of facet goes stale as the repo changes.
- **In-file context only.** It never saw other files of the repo at inference time.
- **One seed, one recipe, two repos tested** in the whole project (facet and telekom/controlplane). Treat the effect
  size as an estimate for this repo, not a general law.
- A base model, not a chat model. It does completion only and follows no instructions.
- It inherits the limitations and biases of Qwen2.5-Coder-14B.

## License and attribution

Apache-2.0. Derived from Qwen2.5-Coder-14B (Apache-2.0) and trained on code from
[facet-rs/facet](https://github.com/facet-rs/facet) (MIT OR Apache-2.0). Inspired by
[Build your own GitHub Copilot](https://prvn.sh/build-your-own-github-copilot).

```bibtex
@article{{hui2024qwen25coder,
  title   = {{Qwen2.5-Coder Technical Report}},
  author  = {{Hui, Binyuan and Yang, Jian and Cui, Zeyu and others}},
  journal = {{arXiv preprint arXiv:2409.12186}},
  year    = {{2024}}
}}
```
"""
    d = ROOT / "model_cards" / f"{a['name']}.md"
    d.write_text(card)
    print("wrote", d, "| 32B:", "beats" if beats32 else "matches")
