#!/usr/bin/env python3
"""Print representative AST holes (code and comment) with their paired random holes from a data build."""

import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clip(s, n=260):
    s = s.replace("\n", "⏎\n")
    return s if len(s) <= n else s[:n] + f" …[+{len(s) - n} chars]"


def main(version, split="train", seed=0):
    d = ROOT / "data" / version
    ast = [json.loads(l) for l in open(d / f"{split}_ast.jsonl")]
    rnd = {r["pair_id"]: r for r in map(json.loads, open(d / f"{split}_random.jsonl"))}
    rng = random.Random(seed)
    code = [e for e in ast if not e["is_comment"]]
    comments = [e for e in ast if e["is_comment"]]
    picks = rng.sample(code, 3) + rng.sample(comments, min(2, len(comments)))
    for e in picks:
        r = rnd[e["pair_id"]]
        print("=" * 100)
        print(f"{e['pair_id']}  crate={e['crate']}  node={e['node_type']}  comment={e['is_comment']}")
        print(f"AST    hole chars {e['hole_chars']} ({len(e['middle'])} chars, {e['tokens']['middle']} tokens); total {e['tokens']['total_with_eos']} tokens")
        print("  prefix tail :", clip(e["prefix"][-120:], 120))
        print("  MIDDLE      :", clip(e["middle"]))
        print("  suffix head :", clip(e["suffix"][:120], 120))
        print(f"RANDOM hole chars {r['hole_chars']} ({len(r['middle'])} chars, {r['tokens']['middle']} tokens); total {r['tokens']['total_with_eos']} tokens")
        print("  prefix tail :", clip(r["prefix"][-120:], 120))
        print("  MIDDLE      :", clip(r["middle"]))
        print("  suffix head :", clip(r["suffix"][:120], 120))


if __name__ == "__main__":
    main(*sys.argv[1:])
