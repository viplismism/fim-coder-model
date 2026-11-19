#!/usr/bin/env python3
"""'Their style' check (prvn.sh setup): exact match on held-out FILES from the same snapshot, with holes cut the same
way as in training. Uses our 5% validation files (never trained on): random-hole set (the style our adapter trained on)
and AST-hole set (closest to prvn.sh). Exact match = same normalisation as the main eval. Paired bootstrap over files.
Usage: python val_style_eval.py <completions_dir> <data_dir> <out.json>
"""
import collections, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_eval import norm_lines

d, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
res = {}
for arm in ("random", "ast"):
    ex = {e["id"]: e for e in map(json.loads, open(data / f"validation_{arm}.jsonl"))}
    C = {n: {c["id"]: c for c in map(json.loads, open(d / f"{n}-val{arm}.completions.jsonl"))} for n in ("qwen14b-base", "qwen14b-ft")}
    em = {n: {i: norm_lines(c["completion"]) == norm_lines(ex[i]["middle"]) for i, c in C[n].items() if "completion" in c} for n in C}
    ids = sorted(set(em["qwen14b-base"]) & set(em["qwen14b-ft"]))
    byf = collections.defaultdict(list)
    for i in ids:
        byf[ex[i]["path"]].append(i)
    fs = sorted(byf); rng = random.Random(20261118); B = 2000; diffs = []
    rate = lambda n, idx: sum(em[n][i] for i in idx) / len(idx)
    for _ in range(B):
        s = [i for f in (fs[rng.randrange(len(fs))] for _ in fs) for i in byf[f]]
        diffs.append(rate("qwen14b-ft", s) - rate("qwen14b-base", s))
    diffs.sort()
    b, f = rate("qwen14b-base", ids), rate("qwen14b-ft", ids)
    res[arm] = {"n": len(ids), "files": len(fs), "base_em": round(b, 4), "ft_em": round(f, 4), "diff_points": round(f - b, 4),
                "relative_gain": round((f - b) / b, 3) if b else None, "ci": [round(diffs[int(.025 * B)], 4), round(diffs[int(.975 * B) - 1], 4)],
                "stopped": {n: sum(C[n][i].get("stopped", False) for i in ids) for n in C}}
    print(arm, res[arm])
out.write_text(json.dumps(res, indent=1) + "\n")
