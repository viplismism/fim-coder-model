#!/usr/bin/env python3
"""Turn test candidates + their execution check into the evaluation set (CPU only, no models).

Packaged for the public release from the inline steps used in the original runs (same logic):
  usable    = the reference file at its commit passes BOTH build and tests (otherwise build+test cannot judge it);
  sensitive = deleting the hole makes the build or the tests fail (only these count for pass@1).
Optionally drop examples whose prompt + 256 new tokens exceed a model's context, for ALL models alike (controlplane
used the DeepSeek-6.7B 16K window so every model saw the same examples; facet kept all and skipped per model).

Usage:
  python -m data.eval_set <candidates.jsonl> <exec.jsonl> <out_eval.jsonl> [--common_context <config.yaml>]
Examples (from the repo root; scripts/facet.sh and scripts/controlplane.sh run these):
  python -m data.eval_set work/eval/facet/test_candidates.jsonl work/eval/facet/exec.jsonl work/eval/facet/eval.jsonl
  python -m data.eval_set work/eval/controlplane/test_candidates.jsonl work/eval/controlplane/exec.jsonl \
      work/eval/controlplane/eval.jsonl --common_context configs/controlplane_deepseek67b.yaml
"""
import json, sys
from pathlib import Path

args = sys.argv[1:]
ctx_cfg = None
if "--common_context" in args:
    k = args.index("--common_context"); ctx_cfg = args[k + 1]; del args[k:k + 2]
cand, exe, out = args
ex = {json.loads(l)["id"]: json.loads(l) for l in open(cand)}
use = [{**ex[r["id"]], "sensitive": r["sensitive"]} for r in map(json.loads, open(exe)) if r["ref_build"] and r["ref_test"]]
use.sort(key=lambda e: e["id"])
dropped = []
if ctx_cfg:
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.model import load_tokenizer
    from core.fim_format import PSM
    cfg = yaml.safe_load(open(ctx_cfg)); psm = PSM(load_tokenizer(cfg), cfg["format"]); lim = cfg["evaluation"]["context_tokens"]
    keep = []
    for e in use:
        n = len(psm.prompt(e["prefix"], e["suffix"]))
        (keep if n + 256 <= lim else dropped).append(e if n + 256 <= lim else {"id": e["id"], "prompt_tokens": n, "sensitive": e["sensitive"]})
    use = keep
    json.dump({"rule": f"prompt + 256 new tokens > {lim} tokens ({ctx_cfg}) -> dropped for ALL models", "dropped": dropped},
              open(Path(out).with_name(Path(out).stem + "_dropped.json"), "w"), indent=1)
Path(out).write_text("".join(json.dumps(e) + "\n" for e in use))
print(f"usable {len(use)}  sensitive {sum(e['sensitive'] for e in use)}  dropped for context {len(dropped)}  -> {out}")
