#!/usr/bin/env python3
"""Hole-size distributions (line counts, aggregate only): synthetic training holes vs real post-cutoff edits.
Explains the AST result: AST training holes are mostly large nodes, real edits are mostly 1-2 lines.
Usage: python -m eval.hole_sizes <dataset_dir> <eval.jsonl> <out.json>
  e.g. python -m eval.hole_sizes work/datasets/rs-facet-qwen14b-v1 work/eval/facet/eval.jsonl results/facet/hole_sizes.json
"""
import collections, json, sys

data, evalf, out = sys.argv[1:4]
B = [("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11-30", 11, 30), ("31+", 31, 10 ** 9)]


def buck(L):
    c = collections.Counter(next(n for n, a, b in B if a <= x <= b) for x in L)
    return {n: round(100 * c[n] / len(L), 1) for n, _, _ in B}


lines = lambda m: m.rstrip("\n").count("\n") + 1
med = lambda L: sorted(L)[len(L) // 2]
tr = [lines(json.loads(l)["middle"]) for l in open(f"{data}/train_ast.jsonl")]
trr = [lines(json.loads(l)["middle"]) for l in open(f"{data}/train_random.jsonl")]
ev = [json.loads(l) for l in open(evalf)]
te = [e["hole_lines"] for e in ev]; tes = [e["hole_lines"] for e in ev if e["sensitive"]]
res = {"note": "line counts of hole texts; aggregate only. Training holes: rs-facet-qwen14b-v1 train split (AST arm; random arm has "
               "identical character lengths). Test holes: facet_eval.jsonl (post-cutoff added blocks, 1-30 lines by construction).",
       "buckets": [n for n, _, _ in B],
       "train_ast": {"n": len(tr), "median_lines": med(tr), "pct": buck(tr)},
       "train_random": {"n": len(trr), "median_lines": med(trr), "pct": buck(trr)},
       "test_all": {"n": len(te), "median_lines": med(te), "pct": buck(te)},
       "test_sensitive": {"n": len(tes), "median_lines": med(tes), "pct": buck(tes)}}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "note"}))
