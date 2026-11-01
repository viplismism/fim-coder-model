#!/usr/bin/env python3
"""Fixed validation subset for evaluation during training: the same pair ids for both arms and every model.

Stratified by package: each validation package gets max(min_per_package, its proportional share) pairs
(capped at what it has), filled by seeded sampling. Writes data/<version>/validation_subset.json.
Usage: python -m fimexp.val_subset --version v --size 128
"""

import argparse, collections, json, math, random

from . import sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--min_per_package", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    cfg = sources.load_config()
    d = sources.ROOT / cfg["data"]["output_dir"] / args.version
    out = d / "validation_subset.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    ast = [json.loads(l) for l in open(d / "validation_ast.jsonl")]
    rnd = {json.loads(l)["pair_id"] for l in open(d / "validation_random.jsonl")}
    assert {r["pair_id"] for r in ast} == rnd, "arms are not matched"
    by_pkg = collections.defaultdict(list)
    for r in ast:
        by_pkg[r["crate"]].append(r["pair_id"])
    total = len(ast)
    quota = {p: min(len(ids), max(args.min_per_package, math.floor(args.size * len(ids) / total))) for p, ids in by_pkg.items()}
    rng = random.Random(args.seed)
    chosen = {p: sorted(rng.sample(sorted(ids), quota[p])) for p, ids in sorted(by_pkg.items())}
    # top up to the requested size from the remaining pairs, proportionally by seeded draw
    rest = sorted(i for p, ids in by_pkg.items() for i in ids if i not in set(chosen[p]))
    extra = rng.sample(rest, max(0, min(len(rest), args.size - sum(map(len, chosen.values())))))
    pair_ids = sorted({i for ids in chosen.values() for i in ids} | set(extra))
    pkg_of = {r["pair_id"]: r["crate"] for r in ast}
    rec = {"version": args.version, "size": len(pair_ids), "seed": args.seed, "min_per_package": args.min_per_package,
           "rule": "per package max(min_per_package, floor(size*share)) capped at available, then seeded top-up; same pair ids for both arms",
           "validation_pairs_total": total, "packages": len(by_pkg),
           "per_package": {p: {"available": len(by_pkg[p]), "selected": sum(1 for i in pair_ids if pkg_of[i] == p)} for p in sorted(by_pkg)},
           "pair_ids": pair_ids}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "pair_ids"}, indent=1))


if __name__ == "__main__":
    main()
