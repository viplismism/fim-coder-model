#!/usr/bin/env python3
"""'Their style' check (prvn.sh setup) for facet: exact match on the 5% held-out validation FILES of the cutoff
snapshot (never trained on), with holes cut the way training holes were cut. Two hole sets: AST holes and random
holes (identical lengths). Three models: base, AST fine-tune, random fine-tune. Same normalisation as the main eval.
Paired bootstrap 95% CIs over files, B=2000.
Usage: python val_style_eval_facet.py <completions_dir> <data_dir> <out.json> [base ast random]
Completions are read from <completions_dir>/<model>-val<arm>.completions.jsonl.
"""
import collections, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_eval import norm_lines

d, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
names = sys.argv[4:7] if len(sys.argv) > 6 else ["qwen14b-base", "facet-ast", "facet-random"]
pairs = [(names[1], names[0]), (names[2], names[0]), (names[1], names[2])]
B = 2000
res = {}
for arm in ("ast", "random"):
    ex = {e["id"]: e for e in map(json.loads, open(data / f"validation_{arm}.jsonl"))}
    C = {n: {c["id"]: c for c in map(json.loads, open(d / f"{n}-val{arm}.completions.jsonl"))} for n in names}
    em = {n: {i: norm_lines(c["completion"]) == norm_lines(ex[i]["middle"]) for i, c in C[n].items() if "completion" in c} for n in names}
    ids = sorted(set.intersection(*(set(em[n]) for n in names)))
    byf = collections.defaultdict(list)
    for i in ids:
        byf[ex[i]["path"]].append(i)
    fs = sorted(byf)
    rate = lambda n, idx: sum(em[n][i] for i in idx) / len(idx)
    rng = random.Random(20261121); diffs = {p: [] for p in pairs}
    for _ in range(B):
        s = [i for f in (fs[rng.randrange(len(fs))] for _ in fs) for i in byf[f]]
        for a, b in pairs:
            diffs[(a, b)].append(rate(a, s) - rate(b, s))
    r = {"n": len(ids), "files": len(fs), "exact_match": {n: round(rate(n, ids), 4) for n in names},
         "hit_256_token_limit": {n: sum(not C[n][i].get("stopped", False) for i in ids) for n in names}, "diffs": {}}
    for (a, b), xs in diffs.items():
        xs.sort(); v = rate(a, ids) - rate(b, ids); rb = rate(b, ids)
        r["diffs"][f"{a} - {b}"] = {"points": round(v, 4), "relative": round(v / rb, 3) if rb else None,
                                    "ci": [round(xs[int(.025 * B)], 4), round(xs[int(.975 * B) - 1], 4)]}
    res[f"{arm}_holes"] = r
    print(arm, json.dumps(r))
out.write_text(json.dumps(res, indent=1) + "\n")
