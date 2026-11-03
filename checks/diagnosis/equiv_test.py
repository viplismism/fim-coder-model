"""Batched vs one-at-a-time greedy generation must give the same completions (validation items, base model)."""
import json, random, sys, time
sys.path.insert(0, ".")
from fimexp import sources
from fimexp.gen_batch import items_validation, generate_batch
from fimexp.model import load_for_scoring, load_tokenizer
from fimexp.psm import PSM
cfg = sources.load_config(); psm = PSM(load_tokenizer(cfg), cfg["format"])
sample = random.Random(1).sample(items_validation(cfg, "rust-fim-v2"), 12)
sample.sort(key=lambda it: len(it["prefix"]) + len(it["suffix"]))
m = load_for_scoring(cfg)
t0 = time.time(); single = [generate_batch(m, psm, [it], 1536)[0] for it in sample]; ts = time.time() - t0
t0 = time.time(); batched = [r for b in range(0, 12, 4) for r in generate_batch(m, psm, sample[b:b + 4], 1536)]; tb = time.time() - t0
same = [a["text"] == b["text"] for a, b in zip(single, batched)]
res = {"items": 12, "identical": sum(same), "single_seconds": round(ts), "batched4_seconds": round(tb),
       "new_tokens": sum(r["new_tokens"] for r in single),
       "first_diff_char": [next((k for k in range(min(len(a["text"]), len(b["text"]))) if a["text"][k] != b["text"][k]), min(len(a["text"]), len(b["text"])))
                           if not s else None for a, b, s in zip(single, batched, same)]}
json.dump(res, open("checks/diagnosis/equiv_test.json", "w"), indent=1); print(res)
