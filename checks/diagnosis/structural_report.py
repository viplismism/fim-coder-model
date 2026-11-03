#!/usr/bin/env python3
"""Structural comparison of base / AST-trained / random-trained completions on VALIDATION pairs.

Metrics per completion: exact match, edit similarity, parse-clean (spliced file has no ERROR/MISSING nodes; every
original validation file parses clean), bracket-balance match (net ()/[]/{} equals the expected answer's), and
compile pass where cargo-checked (subset). Buckets: hole set x boundary type. Paired bootstrap 95% CIs (examples).
Usage: python checks/diagnosis/structural_report.py results/val_structural/gen
"""

import collections, json, random, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "checks" / "diagnosis"))
from diagnose import FileInfo, line_class, node_class  # noqa: E402
from fimexp import sources  # noqa: E402
from fimexp.structural import error_count, score_completion  # noqa: E402

MODELS = ("base", "ast", "random")
PAIRS = (("ast", "random"), ("ast", "base"), ("random", "base"))
B = 2000


def boot(vals_a, vals_b, rng):
    n = len(vals_a)
    d = statistics.fmean(vals_a) - statistics.fmean(vals_b)
    bs = sorted(statistics.fmean(vals_a[i] - vals_b[i] for i in (rng.randrange(n) for _ in range(n))) for _ in range(B))
    lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
    return {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "verdict": "higher" if lo > 0 else "lower" if hi < 0 else "no clear difference"}


def main():
    gen = Path(sys.argv[1])
    cfg = sources.load_config(); reth = ROOT / cfg["sources"]["reth"]["path"]
    items = [json.loads(l) for l in open(gen / "items.jsonl")]
    outs = {m: {r["id"]: r for r in map(json.loads, open(gen / f"{m}.jsonl"))} for m in MODELS}
    comp = {}
    if (gen / "compile.jsonl").exists():
        for l in open(gen / "compile.jsonl"):
            r = json.loads(l); comp[(r["id"], r["model"])] = r["result"] == "pass"
    files, orig_err = {}, {}
    rows = []
    for it in items:
        if not all(it["id"] in outs[m] for m in MODELS):
            continue
        if it["path"] not in files:
            files[it["path"]] = sources.read_source(reth / it["path"])
            orig_err[it["path"]] = error_count(files[it["path"]])
        f = FileInfo(files[it["path"]]).features(*it["hole_chars"])
        row = {"id": it["id"], "set": it["set"], "line": line_class(f), "node": node_class(f), "m": {}}
        for m in MODELS:
            s = score_completion(files[it["path"]], it["hole_chars"], it["middle"], outs[m][it["id"]]["text"], orig_err[it["path"]])
            s["stopped"] = outs[m][it["id"]]["stopped"]
            if (it["id"], m) in comp:
                s["compile_pass"] = comp[(it["id"], m)]
            row["m"][m] = s
        rows.append(row)
    rng = random.Random(0)
    metrics = ("exact_match", "edit_similarity", "parse_clean", "bracket_balance_match", "stopped", "compile_pass")
    report = {"scope": "validation only", "items": len(rows), "original_files_with_parse_errors": sum(1 for v in orig_err.values() if v),
              "buckets": {}}
    def bucket(name, rs):
        out = {"n": len(rs)}
        for met in metrics:
            sub = [r for r in rs if all(met in r["m"][m] for m in MODELS)]
            if len(sub) < 5:
                continue
            v = {m: [float(r["m"][m][met]) for r in sub] for m in MODELS}
            out[met] = {"n": len(sub), **{m: round(statistics.fmean(v[m]), 4) for m in MODELS},
                        **{f"{a}-{b}": boot(v[a], v[b], rng) for a, b in PAIRS}}
        report["buckets"][name] = out
    for s in ("reth_ast", "reth_random"):
        rs = [r for r in rows if r["set"] == s]
        bucket(f"{s} | all", rs)
        for key in ("line", "node"):
            for g in sorted({r[key] for r in rs}):
                bucket(f"{s} | {g}", [r for r in rs if r[key] == g])
    (gen / "structural_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"items scored: {len(rows)}; original files with parse errors: {report['original_files_with_parse_errors']}")
    for name, b in report["buckets"].items():
        print(f"\n== {name}  (n={b['n']})")
        for met in metrics:
            if met in b:
                x = b[met]; ar = x["ast-random"]; ab = x["ast-base"]
                print(f"   {met:22s} n={x['n']:4d}  base {x['base']:.3f}  ast {x['ast']:.3f}  random {x['random']:.3f} | "
                      f"AST-random {ar['diff']:+.3f} [{ar['ci95'][0]:+.3f},{ar['ci95'][1]:+.3f}] {ar['verdict']:19s} | AST-base {ab['diff']:+.3f} {ab['verdict']}")


if __name__ == "__main__":
    main()
